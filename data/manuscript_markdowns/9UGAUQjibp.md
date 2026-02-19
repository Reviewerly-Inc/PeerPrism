Under review as a conference paper at ICLR 2024

QUANTIZED LOCAL INDEPENDENCE DISCOVERY FOR
FINE-GRAINED CAUSAL DYNAMICS LEARNING IN RE-
INFORCEMENT LEARNING

Anonymous authors
Paper under double-blind review

ABSTRACT

Incorporating causal relationships between the variables into dynamics learning
has emerged as a promising approach to enhance robustness and generalization
in reinforcement learning (RL). Recent studies have focused on examining con-
ditional independences and leveraging only relevant state and action variables
for prediction. However, such approaches tend to overlook local independence
relationships that hold under certain circumstances referred as event. In this work,
we present a theoretically-grounded and practical approach to dynamics learning
which discovers such meaningful events and infers fine-grained causal relation-
ships. The key idea is to learn a discrete latent variable that represents the pair
of event and causal relationships specific to the event via vector quantization. As
a result, our method provides a fine-grained understanding of the dynamics by
capturing event-specific causal relationships, leading to improved robustness and
generalization in RL. Experimental results demonstrate that our method is more
robust to unseen states and generalizes well to downstream tasks compared to prior
approaches. In addition, we find that our method successfully identifies meaningful
events and recovers event-specific causal relationships.

1

INTRODUCTION

Model-based reinforcement learning (MBRL) has showcased its capability of solving various sequen-
tial decision making problems (Kaiser et al., 2020; Schrittwieser et al., 2020). Since learning accurate
and robust dynamics model is crucial in MBRL, recent works incorporate the causal relationships
between the variables into dynamics learning (Wang et al., 2022; Ding et al., 2022). Unlike the
traditional dense models that employ the whole state and action variables to predict the future state,
causal dynamics models utilize only relevant variables by examining conditional independences. As a
result, they are more robust to spurious correlations and generalize well to unseen states by discarding
unnecessary dependencies.

Our motivation stems from the observation that the dependencies between the variables often exist
only under certain circumstances in many practical scenarios. For instance, in the context of
autonomous driving, a lane change is contingent on the absence of nearby cars within a specific
distance range. Thus, it is crucial for autonomous vehicles to recognize and understand circumstances
in which lane changes do or do not affect other vehicles. Our hypothesis is that the agent capable of
reasoning such fine-grained causal relationships would generalize well to downstream tasks.

In this work, we aim to incorporate local independence relationship between the variables, which holds
under certain contexts but does not hold in general (Boutilier et al., 2013), into dynamics modeling
for improving robustness and generalization in MBRL. Unfortunately, prior causal dynamics models
examining conditional independences are not capable of harnessing them. An alternative way is to
estimate variables dependencies for each individual sample (Pitis et al., 2020; Hwang et al., 2023).
However, such sample-specific approaches do not explicitly capture meaningful contexts that exhibit
fine-grained causal relationships, making them prone to overfitting and less robust on unseen states.

Contribution. We present a new causal dynamics model that (i) decomposes the data domain
into subgroups which we call events, (ii) discovers local independences under each event, and (iii)
employs only locally relevant variables for prediction (Fig. 1). Clearly, it is crucial to discover

1

Under review as a conference paper at ICLR 2024

Figure 1: Comparison of different types of dynamics models. (a) Dense models employ the whole
state and action variables for prediction. (b) Causal models examine conditional independences
to discard unnecessary dependencies (red arrows in (a)). (c) Sample-specific approaches estimate
variable dependencies on a per-sample basis. (d) Our model decomposes the data domain and infers
fine-grained causal relationships on each event to use only locally relevant variables for prediction.

meaningful context for robust and fine-grained dynamics modeling. For this, we formulate the
problem of finding a decomposition that maximizes the regularized maximum likelihood score and
show that the optimal decomposition identifies a meaningful context that exhibits fine-grained causal
relationships. A main challenge is that this involves three nested subtasks: discovering decomposition,
examining local independences, and learning dynamics model. To this end, we propose a practical
gradient-based method to learn a discrete latent codebook utilizing vector quantization, which enables
the joint optimization differentiable, allowing efficient end-to-end training (Fig. 2). As a result, our
method incorporates fine-grained causal relationships into dynamics modeling, leading to improved
robustness in MBRL over prior causal dynamics models.

We empirically validate the effectiveness of our method on both discrete and continuous control envi-
ronments. For the evaluation, we measure the performance of dynamics models on the downstream
tasks that require fine-grained causal reasoning. Experimental results demonstrate the effectiveness
of our method for fine-grained causal reasoning which improves robustness and generalization in
MBRL. Detailed analysis of our method shows that it successfully discovers meaningful contexts and
recovers fine-grained causal relationships.

2 PRELIMINARIES

We first briefly introduce the notations and terminologies used throughout the paper. Then, we
examine related works on causal dynamics learning for RL and fine-grained causal relationships.

2.1 BACKGROUND

Structural causal model. We adopt a framework of a structural causal model (SCM) (Pearl, 2009) to
understand the relationship among variables in transition dynamics. An SCM
is defined as a tuple
(cid:10)V, U, F, P (U)(cid:11), where V =
is a set of endogenous variables and U is a set of
X1,
{
exogenous variables. A set of functions F =
determine how each variable is generated;
Xj = fj(P a(j), Uj) where P a(j)
induces
⊆
= (V, E), i.e., a causal graph (CG) (Peters et al., 2017), where
a directed acyclic graph (DAG)
V =
E
denotes a direct causal relationship from Xi to Xj. An SCM and its corresponding causal graph
entail the conditional independence relationship of each variable (namely, local Markov property):
Xi

V are the set of nodes and edges, respectively. Each edge (i, j)

P a(Xi), where N D(Xi) is a non-descendant of Xi.

· · ·
is parents of Xj and Uj

1, . . . , d
{

U. An SCM

N D(Xi)

and E

}
f1,

, Xd

{
Xj

, fd

M

M

· · ·

\ {

V

⊆

×

⊆

∈

G

V

}

}

}

⊥⊥

|

Factored Markov Decision Process. A Markov Decision Process (MDP) (Sutton & Barto, 2018) is
defined as a tuple
) is
is a state space,
S
a transition dynamics, r is a reward function, and γ is a discount factor. We consider a factored MDP
N and
(Kearns & Koller, 1999) where the state and action spaces are factorized as

is an action space, T :

S × A → P

, T, r, γ

where

⟨S

A

A

=

S

(

⟩

,

S

1
S

× · · · × S

2

(a)                   (b)                         (c)                                                        (d)<latexit sha1_base64="m9iHAv9KiC4kj6ms0spC2L8VFQg=">AAACBXicbVDLSsNAFJ34rPUVdamLYBFclUTqY1lx47KifUATymQ6aYdOJmHmRighGzf+ihsXirj1H9z5N07aINp64MLhnHu59x4/5kyBbX8ZC4tLyyurpbXy+sbm1ra5s9tSUSIJbZKIR7LjY0U5E7QJDDjtxJLi0Oe07Y+ucr99T6VikbiDcUy9EA8ECxjBoKWeeeCGGIYE8/Q2c4GFVP0Il1nPrNhVewJrnjgFqaACjZ756fYjkoRUAOFYqa5jx+ClWAIjnGZlN1E0xmSEB7SrqcB6n5dOvsisI630rSCSugRYE/X3RIpDpcahrzvzE9Wsl4v/ed0EggsvZSJOgAoyXRQk3ILIyiOx+kxSAnysCSaS6VstMsQSE9DBlXUIzuzL86R1UnXOqqc3tUq9VsRRQvvoEB0jB52jOrpGDdREBD2gJ/SCXo1H49l4M96nrQtGMbOH/sD4+AZ9mJkt</latexit>S⇥AUnder review as a conference paper at ICLR 2024

=

1
A

A
× · · · × A
s, a) where s = (s1,

M , and a single-step transition dynamics is factorized as p(s′
, aM ).

, sN ) and a = (a1,

· · ·

· · ·

s, a) = (cid:81)

j p(s′j |

|

Assumptions and notations. We are concerned with an SCM associated with the transition dynamics
in a factored MDP where we assume that states are fully observable. To properly identify the causal
relationships in MBRL, we make assumptions standard in the field (Ding et al., 2022; Wang et al.,
2021; 2022; Seitzer et al., 2021; Pitis et al., 2020; 2022), namely, Markov property (Pearl, 2009),
faithfulness (Peters et al., 2017), causal sufficiency (Spirtes et al., 2000), and that causal connections
t + 1). Throughout the paper, a causal graph
only appear within consecutive time steps (i.e., t
Y and the set of edges E
Y, where
= (V, E) consists of the set of nodes V = X
. P a(j) denotes parent variables of S′j.
, S′N }
S′1,
{

G
X =
With these assumptions, the conditional independences

and Y =

, SN , A1,

→
∪
· · ·

S1,
{

, AM

· · ·

· · ·

X

⊆

×

}

S′j ⊥⊥

X

\

P a(j)

|

P a(j)

(1)

entailed by the causal graph faithfully represent the causal relationships between the variables and
s, a) = (cid:81)
the transition dynamics is factorized as p(s′

s, a) = (cid:81)

P a(j)).

|

j p(s′j |

j p(s′j |

Dynamics modeling. Traditional dynamics models use the whole state and action variables to predict
the future state, i.e., modeling (cid:81)
s, a). Prior causal dynamics models (Wang et al., 2021;
2022; Ding et al., 2022) examine conditional independences to recover causal relationships and
employ only parent variables for prediction, i.e., modeling (cid:81)
P a(j)). Consequently, causal
dynamics models are more robust to unseen states by discarding unnecessary dependencies. In
this work, we infer fine-grained causal relationships by discovering local independences and use
potentially fewer dependencies for dynamics modeling, as shown in Fig. 1.

j p(s′j |

j p(s′j |

2.2 RELATED WORK

Causal dynamics models in RL. There is a growing body of literature on the intersection of causality
and RL (De Haan et al., 2019; Buesing et al., 2019; Zhang et al., 2020a; Sontakke et al., 2021;
Sch¨olkopf et al., 2021; Zholus et al., 2022; Zhang et al., 2020b). One focus is causal dynamics
learning, which aims to infer the causal structure of the underlying transition dynamics (Li et al.,
2020; Yao et al., 2022; Bongers et al., 2018; Wang et al., 2022; Ding et al., 2022; Feng et al., 2022;
Huang et al., 2022) (more broad literature of causal reasoning in RL is discussed in Appendix A.1).
Given the explicit state and action variables in factored MDP, recent works utilize gradient-based
causal discovery algorithm (Wang et al., 2021; Brouillard et al., 2020), conditional independence
tests (Ding et al., 2022), or conditional mutual information (Wang et al., 2022) to infer the causal
graph and train the dynamics model with the inferred causal graph by using only relevant variables
for prediction. In contrast, our method infers fine-grained causal relationships by discovering local
independences. Thus, our approach provides a more detailed understanding of the dynamics, leading
to improved robustness and generalization over the prior causal dynamics models.

Discovering fine-grained causal relationships. The fine-grained causal relationships have been
utilized to improve RL performance in various ways, e.g., with data augmentation (Pitis et al., 2022),
efficient planning (Hoey et al., 1999; Chitnis et al., 2021), or exploration (Seitzer et al., 2021).
Previous works exploited prior knowledge of them (Pitis et al., 2022), or leveraged the true dynamics
model explicitly (Chitnis et al., 2021). Without those prior information, Pitis et al. (2020) devised an
transformer-based model to estimate the variable dependencies for each sample by using attention
score. Another line of work learn sparse and modular dynamics (Goyal et al., 2021c;b;a), which can
be viewed as an implicit approach to discovering local independence relationships. In the field of
causality, local independence relationship has been widely studied especially for discrete variables,
e.g., context-specific independence (Boutilier et al., 2013; Zhang & Poole, 1999; Poole, 1998; Dal
et al., 2018; Tikka et al., 2019) (see Appendix A.2 for the background on local independence). Re-
cently, NCD (Hwang et al., 2023) proposed a gradient-based method to discover local independences
allowing continuous variables. While it also infers local independences on a per-sample basis, our
method infers local independences per event, i.e., subgroup of the data domain, which helps prevent
overfitting to individual samples and allows more robust causal modeling.

3

Under review as a conference paper at ICLR 2024

3 FINE-GRAINED CAUSAL DYNAMICS LEARNING

We first describe a brief background on the local independences and local causal graph that represents
the fine-grained causal relationships (Sec. 3.1). We then formulate a problem of finding the optimal
decomposition and describe its implications (Sec. 3.2). As a practical approach, we present our
proposed causal dynamics model that discovers decomposition and event-specific causal relationships
with vector quantization, which enables joint differentiable optimization (Sec. 3.3). Finally, we
provide a theoretical analysis of our approach to identifying the meaningful context that exhibits
fine-grained causal relationships (Sec. 3.4).

3.1 LOCAL INDEPENDENCE AND LOCAL CAUSAL GRAPH

We first describe how local independence provides a way to understand fine-grained causal relation-
ships between the variables. Analogous to the conditional independence in Eq. (1) explaining the
causal relationship between the variables, local independence, which is written as:

),

,

E

E

(2)

=

P a(j) is a
)
E
and the rest of the parent variables become

S × A
holds,1 implies that only P a(j;

and P a(j;

⊆

E

E

)

S′j ⊥⊥
is a subset of the joint state and action space

P a(j;

E

\

P a(j;

)

|

X

E ⊆ X

where
X
minimal subset of P a(j) in which the local independence on
are locally relevant variables for prediction on event
redundant.
Definition 1 (Local Causal Graph). Local causal graph (LCG) on
E
E

∈
Local causal graph
relationships under the event
GX
our goal is to find important contexts that entail a fine-grained causal relationship.
Proposition 1 (Monotonicity). Let

)
}
is a subgraph of the causal graph
G
⊊
G

(i, j)
{

. Clearly,

GE ⊆ G

. Then,

. Also,

P a(j;

GE

=

=

G

E

E

E

i

|

.

.

which represents fine-grained causal
, and
does not always hold on any

E

E ⊆ X

is

GE

= (V, E

E

) where

F ⊆ E

GF ⊆ GE

As the event we focus on becomes more specific (i.e.,
(i.e.,
context which is more likely (i.e., large p(

), finer-grained relationships may arise
F ⊆ E
), but it also becomes less likely to happen. Therefore, it is important to capture the

)), and more meaningful (i.e., sparse

GF ⊆ GE

).

GE

E

3.2 SCORE FOR THE DECOMPOSITION AND GRAPHS

where K is a small number which is a
We consider a decomposition
hyperparameter of our model. By decomposing the domain into a few subgroups, we aim to capture
meaningful contexts that render sparse dependencies for robust and fine-grained dynamics modeling.
It is worth noting that such events are not given as prior information.

K
z=1 of the domain

z
{E

X

}

For now, let us consider an arbitrary decomposition
the decomposition, defined as Z = z if (s, a)
we denote P a(j;
z as
P a(j, z),
dynamics for each S′j can be written as:

K
z=1. We define a variable Z representing
z
{E
}
z for all z
[K] (Hwang et al., 2023). For brevity,
∈ E
z) as P a(j, z) and
z. Each local independence S′j ⊥⊥
E
G
z is then equivalently written as S′j ⊥⊥
|
E

|
P a(j, z), Z = z. The transition

P a(j, z)

P a(j, z)

GE

X

X

∈

\

\

p(s′j |

s, a) =

(cid:88)

z

p(s′j |

s, a, z)p(z

s, a) =

|

(cid:88)

z

p(s′j |

P a(j, z), z)p(z

s, a),

|

(3)

s, a) = 1 if (s, a)

where p(z
dynamics modeling, i.e., employing only locally relevant variables P a(j, z) for each
consider the following regularized maximum likelihood score:

z otherwise 0. This illustrates our approach to fine-grained
z. We now
E

∈ E

|

(

S

z,

{G

z=1) := sup E (cid:2)log ˆp(s′
K
}

z
E

|

s, a;

z,

{G

)

z
E

}

−

λ

z
|G

|

(cid:3) ,

(4)

K
z=1 is the decomposition,

z, and the dynamics model ˆp uses
E
z. It is worth noting that due to the nature of factored MDP where the causal graph is
E

z is the graph on each

G

where

z
{E
}
z for each

G

1We provide a formal definition and detailed background of local independence in Appendix B.1.

4

Under review as a conference paper at ICLR 2024

Figure 2: Overall framework. (a) For each sample (s, a), our method infers the event to which the
sample belongs through quantization, and the corresponding event-specific local causal graph (LCG)
that represents fine-grained causal relationships. (b) Dynamics model is trained to predict the next
state using only relevant variables based on the inferred LCG.

directed bipartite, each Markov equivalence class contains a single unique causal graph. Given this
background, the causal graph is uniquely identifiable with oracle conditional independence test (Ding
et al., 2022) or score maximization (Huang et al., 2018; Brouillard et al., 2020).

z
{E

ˆ
z
G
{

K
z=1 ∈
}
z is true LCG on corresponding
where

K
K
z=1 be the graphs on each
z=1 be the arbitrary decomposition. Let
}
}
K
z=1). With the Assumptions 1 to 4,
argmax
}
z for small enough λ > 0. In particular, if K = 1, then
E

Proposition 2. Let
z that maximizes the score:
E
each ˆ
G
ˆ
=
G
G
If K = 1, this degenerates to the prior score-based approach and would yield
. On the other hand,
any arbitrary decomposition of K > 1 also does not always provide a fine-grained understanding, e.g.,
for all z in the worst case. Thus, we aim to discover the decomposition (and corresponding

is the ground truth causal graph.

(
{G

z
E

} S

z
{G

z,

G

G

{

z =

ˆ
z
G

G
LCGs) that maximizes the score.

G

z

z

E

(

z,

{G

z,

} S

argmax

z=1). Then, each ˆ
K
z
G
}
E
K
z=1 be the arbitrary decomposition and
}
(cid:3)

{G
{F
z. Then, with the Assumptions 1 to 5, E(cid:2)
ˆ
z
G
|
F

z, ˆ
ˆ
K
Proposition 3. Let
z
z=1 ∈
{
}
E
G
on ˆ
[K]. Also, let
z for all z
∈
E
corresponding true LCGs on each
for small enough λ > 0.
This states that the decomposition that maximizes the score is optimal in terms of E (cid:2)
(cid:80)
causal relationships (i.e., sparse
approach to find such decomposition for robust and fine-grained dynamics modeling in MBRL.

(cid:3) =
|
, which implies that this captures the meaningful events that exhibit fine-grained
|
z)). We now proceed to describe our practical
z with large p(
E

z is the true LCG
K
z=1 be the
(cid:3) holds

z
}
{G
E (cid:2)
z
|G
≤

z p(

z
|G

z
|G

z)

G

E

|

|

3.3 CAUSAL DYNAMICS LEARNING WITH QUANTIZED LOCAL INDEPENDENCE DISCOVERY

z,

{

{G

z
E

z,

} S

z
E

(
{G

z, ˆ
ˆ
z
E
G

K
z=1 ∈
}

As described above, the decomposition and corresponding graphs that maximize the score (Eq. (4)),
K
z=1), provide a fine-grained understanding of the
i.e.,
argmax
}
) which captures meaningful contexts, (ii)
dynamics. Our goal is to (i) find decomposition (i.e.,
}
discover locally relevant variables P a(j, z) on each event (i.e.,
), and (iii) train the dynamics
model ˆp(s′
s, a) using them. However, this involves three nested subtasks, and naive optimization
with respect to decomposition is generally intractable. To this end, we propose a practical gradient-
based method which allows efficient joint optimization of three objectives. Our key idea is to
z), i.e., the pair of
learn a discrete latent codebook C =
E
event and corresponding graph, by utilizing vector quantization. The training of the codebook is
differentiable and can be jointly trained with the dynamics model ˆp, resolving the challenging task of
joint optimization of three objectives. The overall framework is illustrated in Fig. 2.

where each code ez represents (

ez
{

z
{G

z
{E

z,

G

}

}

|

Discrete latent codebook representing the decomposition. First, with the encoder genc, each
sample (s, a) is encoded into a latent embedding h, which is then quantized to the nearest prototype
vector (i.e., code) in the codebook C =
, eK

, following (Van Den Oord et al., 2017) as:

e1,

{

· · ·

}

e = ez, where z = argmin

h

[K] ∥

ej

2.

∥

−

(5)

j

∈

5

EncoderDecoder(a) Local Causal Graph Inference(b) Masked Prediction with LCGState or Action VariablesMasked VariablesDynamics Model<latexit sha1_base64="7o47/186CuYTcMwNodu2vFO8Xng=">AAAB7XicbVDLSgNBEOyNrxhfUY9eBoMQQZZd8XUMePEYwTwgWcLsZDYZMzuzzMwKYck/ePGgiFf/x5t/4yTZgyYWNBRV3XR3hQln2njet1NYWV1b3yhulra2d3b3yvsHTS1TRWiDSC5VO8SaciZowzDDaTtRFMchp61wdDv1W09UaSbFgxknNIjxQLCIEWys1KzqM4RPe+WK53ozoGXi56QCOeq98le3L0kaU2EIx1p3fC8xQYaVYYTTSambappgMsID2rFU4JjqIJtdO0EnVumjSCpbwqCZ+nsiw7HW4zi0nTE2Q73oTcX/vE5qopsgYyJJDRVkvihKOTISTV9HfaYoMXxsCSaK2VsRGWKFibEBlWwI/uLLy6R57vpX7uX9RaXm5nEU4QiOoQo+XEMN7qAODSDwCM/wCm+OdF6cd+dj3lpw8plD+APn8wcZR44d</latexit>(s,a)<latexit sha1_base64="rLydIPTJEtYRfoZpDbg2DKGKs/w=">AAAB6XicbVDLSgNBEOyNrxhfUY9eBoPoKeyKr2PAi8co5gHJEmYnvcmQ2dllZlYIS/7AiwdFvPpH3vwbJ8keNLGgoajqprsrSATXxnW/ncLK6tr6RnGztLW9s7tX3j9o6jhVDBssFrFqB1Sj4BIbhhuB7UQhjQKBrWB0O/VbT6g0j+WjGSfoR3QgecgZNVZ60Ke9csWtujOQZeLlpAI56r3yV7cfszRCaZigWnc8NzF+RpXhTOCk1E01JpSN6AA7lkoaofaz2aUTcmKVPgljZUsaMlN/T2Q00nocBbYzomaoF72p+J/XSU1442dcJqlByeaLwlQQE5Pp26TPFTIjxpZQpri9lbAhVZQZG07JhuAtvrxMmudV76p6eX9RqVXzOIpwBMdwBh5cQw3uoA4NYBDCM7zCmzNyXpx352PeWnDymUP4A+fzBzxjjR4=</latexit>s0<latexit sha1_base64="7o47/186CuYTcMwNodu2vFO8Xng=">AAAB7XicbVDLSgNBEOyNrxhfUY9eBoMQQZZd8XUMePEYwTwgWcLsZDYZMzuzzMwKYck/ePGgiFf/x5t/4yTZgyYWNBRV3XR3hQln2njet1NYWV1b3yhulra2d3b3yvsHTS1TRWiDSC5VO8SaciZowzDDaTtRFMchp61wdDv1W09UaSbFgxknNIjxQLCIEWys1KzqM4RPe+WK53ozoGXi56QCOeq98le3L0kaU2EIx1p3fC8xQYaVYYTTSambappgMsID2rFU4JjqIJtdO0EnVumjSCpbwqCZ+nsiw7HW4zi0nTE2Q73oTcX/vE5qopsgYyJJDRVkvihKOTISTV9HfaYoMXxsCSaK2VsRGWKFibEBlWwI/uLLy6R57vpX7uX9RaXm5nEU4QiOoQo+XEMN7qAODSDwCM/wCm+OdF6cd+dj3lpw8plD+APn8wcZR44d</latexit>(s,a)CodebookEvent-Speciﬁc LCGsQuantizationUnder review as a conference paper at ICLR 2024

The quantization entails the decomposition of
since each sample corresponds to exactly one
X
of the latent codes. In other words, the discrete latent codebook represents the decomposition as
s, a) in
e = ez
s, a) = 1 if e = ez and otherwise 0, i.e., determines the event to which the sample

[K]. The quantization corresponds to the term p(z

for all z

z =

(s, a)
E
|
{
Eq. (3) as: p(z
|
belongs. Thus, the codebook C =

K
z=1 serves as a proxy for decomposition

K
z=1.

∈

}

|

z

ez
{

}

{E

}

Discrete latent codebook representing the local independences. Each code ez is then decoded to
(N +M )
an adjacency matrix Az
z. In particular, the
0, 1
}
output of the decoder gdec is the parameters of Bernoulli distributions from which adjacency matrix
(cid:3). To properly backpropagate gradients, we adopt Gumbel-Softmax
gdec(e) = (cid:2)pij
is sampled: A
reparametrization trick (Jang et al., 2016; Maddison et al., 2016).

N that represents the inferred graph

∈ {

∼

G

×

Dynamics model learning. The dynamics model employs only relevant variables for prediction
s, a; A) = (cid:80)
with respect to the adjacency matrix A (Fig. 2 (b)). Specifically, log ˆp(s′
j log ˆp(s′j |
|
(N +M ) indicates whether
s, a; Aj), where Aj is the j-th column of A. Each entry of Aj
0, 1
}
the corresponding state or action variable will be used or not to determine the next state s′j. This
s, a; Aj), we
corresponds to the term p(s′j |
mask out the features of unused variables (Brouillard et al., 2020).

P a(j, z), z) in Eq. (3). For the implementation of ˆp(s′j |

∈ {

Training objective. We employ a regularization loss λ
hyperparameter. The masked prediction loss with the regularization is as follows:

1 to induce a sparse LCG, where λ is a
∥

· ∥

A

(6)

(7)

−
To update the codebook, we use a quantization loss (Van Den Oord et al., 2017):

· ∥

L

∥

|

pred =

log ˆp(s′

s, a; A) + λ

A

1.

quant =

L

sg [h]
∥

e
∥

−

2
2 + β

h

· ∥

−

sg [e]

2
2,
∥

where sg [
] is a stop-gradient operator and β is a hyperparameter. The first term is the codebook loss
·
which moves each code toward the center of the embeddings assigned to it. The second term is the
commitment loss which encourages the quantization encoder outputs the embeddings close to the
prototype vectors. The resulting training objective is

total =

pred +

quant.

L

L

L

3.4

IDENTIFIABILITY AND DISCUSSIONS

(

}

E

G

S

L

L

z,

z,

⊊

{G

z
E

where

K
z=1) in Eq. (4) and

Theorem 1 (Identifiability). Let
. Suppose

So far, we have described how our method learns the decomposition and LCGs through the discrete
latent codebook C as a proxy where each code ez corresponds to the pair of event and graph (
z),
and joint training of the dynamics model ˆp and the codebook C. Considering that
pred corresponds
to the score
quant is a mean squared error in the latent space which
can be minimized to 0, our method is a practical approach to the score maximization which allows
efficient end-to-end training. Finally, we show the identifiability that the optimal decomposition that
maximizes the score identifies a meaningful context that exhibits fine-grained causal relationships.
K
z=1) and K > 1. Let
}
c. Then,
for any
ˆ
i) = 0,
and p(
E

D
with the Assumptions 1 to 5, there exists I ⊊ [K] such that (i) (cid:83)
(ii) ˆ
G

G
which
It states that the optimal decomposition
D
exhibits fine-grained causal relationships, in the sense that events
almost surely
where I ⊊ [K]. Thm. 1 implies that any choice of K > 1 would lead to the identification of
meaningful context. In our method, the codebook size K represents to the size of the decomposition,
and in practice, we found that our method works reasonably well for any choice of K > 1. Omitted
proofs are provided in Appendix B.2.

for all z /
∈
[K] would discover the meaningful context
∈
ˆ
z
E
{

K
z=1 ∈
}
GD

z, ˆ
ˆ
z
E
G
{
=
GF

z
E
G
⊆ D

(
{G
GF
ˆ
i
E

F ⊆ D
(cid:83)
I
i
∈

I and ˆ
G

{G
F ⊆ D

z,
z
} S
E
, and

I identify

for all z

argmax

for any

ˆ
z
E
{

z,
=

z =

z =

D \

GD

GD

z
}

}z

I.

⊊

D

X

∈

G

∈

∈

I

i

4 EXPERIMENTS

In this section, we evaluate our method to examine the following questions: (1) Does our dynamics
model improves robustness and generalization of MBRL? (Tables 1 and 2); (2) Does our method
capture a meaningful context and fine-grained causal relationships? (Figs. 4 and 5); and (3) How
does the choice of K affect performance? (Table 3)

6

Under review as a conference paper at ICLR 2024

(a) CHEMICAL

(b) MAGNETIC

Figure 3: Illustrations for each environment. (a) In Chemical, colors change by the action according
to the underlying causal graph. (b) In Magnetic, the red object exhibits magnetism. (b-left) The box
attracts the ball via magnetic force. (b-right) The box does not have an influence on the ball.

4.1 EXPERIMENTAL SETUP

The environments are designed to exhibit fine-grained causal relationships on a particular context, and
explicit state variables are given as the observation to the agent (e.g., position, velocity), following
prior works (Ding et al., 2022; Wang et al., 2022; Seitzer et al., 2021; Pitis et al., 2020; 2022).
Detailed descriptions for each environment and setup are provided in Appendix C.1.

4.1.1 ENVIRONMENTS

Chemical. In the Chemical (Ke et al., 2021) environment, there are 10 nodes where each node is
colored with one of 5 colors, and an action is setting the color of a node. According to the underlying
causal graph, an action changes the colors of the intervened object’s descendants as depicted in
Fig. 3(a). The task is to match the colors of each node to the given target. We design two settings,
named full-fork and full-chain: the underlying causal graph is both full, and the local causal graph
is fork and chain, respectively. For example, in full-fork, local causal graph fork corresponds to the
context where the color of the root node is red. In other words, for each node, all other parent nodes
except the root become irrelevant according to the particular color of the root node. During the test,
the root color is set to activate local causal graph. Here, the agent receives a noisy observation for
some nodes (except the root) and the task is to match the colors of other observable nodes. The agent
capable of fine-grained causal reasoning would generalize well since corrupted nodes are locally
spurious to infer the colors of other nodes, as depicted in Appendix C.1 (Fig. 6).

Magnetic. We design the robot arm manipulation environment based on the Robosuite framework
(Zhu et al., 2020). There is a moving ball and a box on the table, colored red or black (Fig. 3(b)).
Red color indicates that the object is magnetic, and attracts the other magnetic object. For example,
when they are both colored red, magnetic force will be applied, and the ball will move toward the
box. Otherwise, the box would have no influence on the ball. The task is to move the robot arm to
reach the ball predicting its trajectory. The color and position of the objects are randomly initialized
for each episode. During the test, one of the objects is black, and the box is located at an unseen
position. Since the position of the box is unnecessary for predicting the movement of the ball under
non-magnetic context, the agent aware of the fine-grained causal relationships would generalize well
to unseen out-of-distribution (OOD) states.

4.1.2 BASELINES

We first consider traditional dense models, i.e., a monolithic network implemented as multi-layer
perceptron (MLP) which approximates the dynamics p(s′
s, a), and a modular network which has a
separate network for each variable: (cid:81)
s, a). In addition, we include a graph neural network
(GNN) (Kipf et al., 2020) and NPS (Goyal et al., 2021a). GNN learns the relational information
between variables and NPS learns sparse and modular dynamics. Causal models, including CDL
(Wang et al., 2022) and GRADER (Ding et al., 2022), infer causal relationships between the variables
for dynamics learning: (cid:81)
P a(j)), utilizing conditional mutual information and conditional
independence test, respectively. We also include an oracle causal model, which leverages the ground
truth (global) causal graph. Finally, we compare to a local causal model, NCD (Hwang et al., 2023),
which infers the variable dependencies for each sample.

j p(s′j |

j p(s′j |

|

7

FullChainForkUnder review as a conference paper at ICLR 2024

Table 1: Average episode reward on training and downstream tasks in each environment. In Chemical,
n denotes the number of noisy nodes in downstream tasks.

Chemical (full-fork)

Chemical (full-chain)

Magnetic

Methods

Train
(n = 0)

Test
(n = 2)

Test
(n = 4)

Test
(n = 6)

Train
(n = 0)

Test
(n = 2)

Test
(n = 4)

Test
(n = 6)

Train

Test

±

±

±

19.00
MLP
18.55
Modular
18.60
GNN
7.71
NPS
±
CDL
18.95
GRADER 18.65
19.64
Oracle
19.30
NCD
19.28
Ours

±

±

±

±

0.83

1.00

1.19

1.22

1.40

0.98

1.18

0.95

0.87

±

0.48

0.70

0.92

0.83

1.33

1.31

0.87

±

±

±

±

6.49
6.05
6.61
5.82
9.37
9.27
7.83
±
10.95
15.27

±

±

1.63

2.53

±

±

0.71

0.50

0.74

0.57

0.40

0.65

0.62

0.63

±

±

±

±

5.93
5.65
6.15
5.75
8.23
8.79
8.04
9.11
±
14.73

±

±

±

1.68

±

±

±

±

±

6.84
6.43
6.95
5.54
9.50
±
10.61
9.66
±
10.32
13.62

1.17

1.00

0.78

0.80

1.18

1.31

±

0.21

0.93

2.56

±

±

±

±

±

17.91
17.37
16.97
8.20
±
17.95
17.71
17.79
18.27
17.22

±

±

±

±

±

0.87

1.63

1.85

0.54

0.83

0.54

0.76

0.27

0.61

0.65

0.63

0.28

1.03

0.55

0.56

0.69

1.52

±

±

±

±

7.39
6.61
6.89
6.92
8.71
8.69
8.47
9.60
±
13.36

±

±

±

3.60

±

0.58

0.55

0.28

0.79

0.38

0.80

0.78

0.23

±

±

±

±

6.63
7.01
6.38
6.88
8.65
8.75
8.85
8.86
±
12.35

±

±

±

3.23

±

±

±

±

6.78
7.04
6.56
6.80
±
10.23
10.14
10.29
10.32
12.00

±

±

±

±

±

0.93

1.07

0.53

0.39

0.50

0.33

0.37

0.37

1.21

±

±

±

±

8.37
8.45
8.53
3.13
8.75
±
-
8.42
8.48
8.52

±

±

±

0.74

0.80

0.83

1.00

0.69

0.86

0.70

0.74

±

±

±

±

0.86
0.88
0.92
0.91
1.10
±
-
0.95
1.31
4.81

±

±

±

0.45

0.52

0.51

0.69

0.67

0.55

0.77

3.01

Table 2: Prediction accuracy on ID (n = 0) and OOD (n = 2, 4, 6) states in Chemical environment.

Setting

MLP

Modular

GNN

NPS

CDL

GRADER

Oracle

NCD

Ours

full
-fork

full
-chain

(n = 0)
(n = 2)
(n = 4)
(n = 6)

(n = 0)
(n = 2)
(n = 4)
(n = 6)

88.31
31.11
30.44
32.39

84.38
28.66
26.52
24.15

1.58

1.69

2.28

1.76

1.31

3.65

4.26

4.17

±

±

±

±

±

±

±

±

89.24
26.53
24.73
26.73

85.92
25.24
24.94
25.09

1.52

3.45

5.61

8.31

1.15

4.68

4.81

5.91

±

±

±

±

±

±

±

±

88.81
36.29
25.80
21.58

85.41
29.22
23.28
20.53

1.44

3.45

3.48

3.44

1.84

3.39

4.98

6.96

±

±

±

±

±

±

±

±

58.34
40.56
26.81
23.02

58.48
38.73
27.69
24.45

2.08

4.61

4.37

4.27

2.81

2.63

4.28

3.84

±

±

±

±

±

±

±

±

89.22
35.59
35.82
42.22

86.85
34.90
36.52
42.06

1.67

1.85

1.40

1.39

1.47

1.59

1.72

1.29

±

±

±

±

±

±

±

±

87.75
37.93
38.94
45.74

84.24
36.82
37.41
43.48

1.64

1.06

1.63

2.25

±

±

±

±

1.22

3.12

2.84

4.14

±

±

±

±

89.63
33.87
36.48
42.47

85.76
34.63
38.31
42.87

1.62

1.34

1.80

0.75

1.56

1.78

2.48

2.08

±

±

±

±

±

±

±

±

90.07
41.60
37.47
42.27

85.63
40.04
37.47
41.19

1.22

5.08

2.13

1.82

1.01

6.21

2.98

1.66

±

±

±

±

±

±

±

±

±

89.46
66.44
58.49
±
49.09

±

1.40

12.22

10.20

4.77

±

±

86.07
60.34
±
56.64
53.29

±

±

1.62

12.10

9.40

6.63

Planning algorithm. For all baselines and our method, we use a model predictive control (MPC)
(Camacho & Alba, 2013) which selects the actions based on the prediction of the learned dynamics
model. Specifically, we use the cross-entropy method (CEM) (Rubinstein & Kroese, 2004), which
iteratively generates and optimizes action sequences.

Implementation. For the implementation of our method, we set the codebook size of K = 16, the
regularization coefficient λ = 0.001, and the commitment coefficient β = 0.25 in all experiments. For
the oracle causal model, we used the same network architecture as ours, to isolate the effect of learning
fine-grained causal relationships. All methods have a similar number of model parameters for a fair
comparison. For the evaluation, we ran 10 test episodes for every 40 training episodes. The results are
averaged over eight different runs. Implementation details are provided in Appendix C.2. Learning
curves for all downstream tasks with additional experimental results are shown in Appendix C.3.

4.2 DOWNSTREAM TASK PERFORMANCE

Table 1 demonstrate the downstream task performance of our method and baselines. While all
the methods show similar performance in training, dense models suffer from OOD states in the
downstream tasks. Causal models are generally more robust compared to dense models, as they
infer the causal graph to discard irrelevant dependencies. Our method significantly outperforms
the baselines in all downstream tasks, which implies that our method is capable of fine-grained
causal reasoning and generalize well to unseen states. To investigate the robustness of our method
in downstream tasks, we examine the prediction accuracy on in-distribution (ID) states, and OOD
states in Chemical environment (Table 2). While all methods show similar prediction accuracy on
ID states, the baselines show a significant performance drop under the presence of noisy variables,
merely above 20% which is an expected accuracy of random guessing. In contrast, our method
consistently outperforms the baselines by a large margin, which implies that it successfully captures
the fine-grained causal relationships. These results demonstrates the effectiveness of our method for
fine-grained causal reasoning which improves robustness and generalization in MBRL.

4.3 DETAILED ANALYSIS

Inferred LCGs. We closely examine the discovered local causal graphs in Chemical (full-fork) to
better understand our model’s behavior (Fig. 4). Recall each code corresponds to the pair of event
and LCG, we observe that all codes are used during training (Fig. 4(a)), but only a few of them are
exploited during the test (Fig. 4(b)). The most commonly inferred LCG on OOD states during the

8

Under review as a conference paper at ICLR 2024

(a)

(b)

(c)

(d)

Figure 4: (a,b) Codebook histogram on (a) ID states during training and (b) OOD states during the test
in Chemical (full-fork). (c) True causal graph of the fork structure. (d) Learned LCG corresponding
to the most used code in (b).

(a)

(b)

(c)

(d)

Figure 5: (a) Causal graph identified by our method in Magnetic. Red boxes indicate locally irrelevant
edges under the non-magnetic event. (b-d) LCGs corresponding to the non-magnetic event inferred
by (b) our method, and NCD on (c) ID and (d) OOD state.

test corresponds to a fork structure, as shown in Figs. 4(c) and 4(d). This implies that our method
successfully identifies the meaningful context that exhibits fine-grained causal relationships.

Methods

Chemical (full-fork)

Table 3: Ablation on codebook size.

Comparison with sample-specific approach. In Fig. 5,
we examine the LCGs to further analyze the improved ro-
bustness of our approach compared to a sample-specific
approach that does not explicitly capture meaningful
events. As shown in Fig. 5(b), our method learns a
proper LCG during training, and the inference is consis-
tent in both ID and OOD states on non-magnetic context.
In contrast, as shown in Figs. 5(c) and 5(d), the inference
of sample-specific approach is inconsistent between ID
and OOD states on the same non-magnetic context. This
is because sample-specific approach infers the variable
dependencies for each sample, and this incurs overfitting which makes the inference on OOD states
inconsistent. As opposed to sample-specific approach, our method learns an LCG for each event,
leading to the robust inference on OOD states.

CDL
Oracle
NCD
Ours (K = 2)
Ours (K = 4)
Ours (K = 8)
Ours (K = 16)
Ours (K = 32)

9.37
7.83
±
10.95
13.44
15.73
14.95
15.27
16.12

9.50
9.66
±
10.32
12.99
12.40
13.42
13.62
14.79

8.23
8.04
9.11
±
12.86
16.50
15.03
14.73
14.35

(n = 4)

(n = 6)

(n = 2)

2.67

1.37

2.53

2.13

1.16

2.56

1.68

2.61

1.43

0.63

1.33

4.13

3.40

5.27

0.40

1.63

0.62

0.87

1.18

5.58

0.93

5.41

2.81

0.21

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

±

±

±

±

±

±

±

±

±

Ablation on the codebook size. We first recall that the codebook size K represents the size of
the decomposition. As shown in Table 3, our method works reasonably well with any choice of K
and consistently outperforms baselines. Finally, as we described earlier, our method with K = 1
degenerates to prior causal dynamics model and cannot capture fine-grained causal relationships.
This is shown in Fig. 5(a) that our method with K = 1 recovers the causal graph including locally
irrelevant edges under the non-magnetic context.

5 CONCLUSION

We presented a novel approach to causal dynamics learning that infers fine-grained causal relation-
ships, improving robustness and generalization of MBRL compared to previous approaches. We show
that the decomposition that maximizes the score identifies the meaningful context existing in the
system. As a practical approach, our method learns a discrete latent variable that represents the pairs
of event and event-specific causal relationships by utilizing vector quantization. Compared to prior
approaches, our method provides a fine-grained understanding of the dynamics and allows robust
causal dynamics modeling by capturing event-specific causal relationships. We discuss limitations
and future works in Appendix C.3.5.

9

Under review as a conference paper at ICLR 2024

REFERENCES

Silvia Acid and Luis M de Campos. Searching for bayesian network structures in the space of
restricted acyclic partially directed graphs. Journal of Artificial Intelligence Research, 18:445–490,
2003.

Elias Bareinboim, Andrew Forney, and Judea Pearl. Bandits with unobserved confounders: A causal

approach. Advances in Neural Information Processing Systems, 28, 2015.

Ioana Bica, Daniel Jarrett, and Mihaela van der Schaar.

Invariant causal imitation learning for
generalizable policies. Advances in Neural Information Processing Systems, 34:3952–3964, 2021.

Stephan Bongers, Tineke Blom, and Joris M Mooij. Causal modeling of dynamical systems. arXiv

preprint arXiv:1803.08784, 2018.

Craig Boutilier, Nir Friedman, Mois´es Goldszmidt, and Daphne Koller. Context-specific independence

in bayesian networks. CoRR, abs/1302.3562, 2013.

Philippe Brouillard, S´ebastien Lachapelle, Alexandre Lacoste, Simon Lacoste-Julien, and Alexandre
Drouin. Differentiable causal discovery from interventional data. arXiv preprint arXiv:2007.01754,
2020.

Lars Buesing, Theophane Weber, Yori Zwols, Nicolas Heess, Sebastien Racaniere, Arthur Guez,
and Jean-Baptiste Lespiau. Woulda, coulda, shoulda: Counterfactually-guided policy search. In
International Conference on Learning Representations, 2019.

Eduardo F Camacho and Carlos Bordons Alba. Model predictive control. Springer science & business

media, 2013.

Rohan Chitnis, Tom Silver, Beomjoon Kim, Leslie Kaelbling, and Tomas Lozano-Perez. Camps:
Learning context-specific abstractions for efficient planning in factored mdps. In Conference on
Robot Learning, pp. 64–79. PMLR, 2021.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of
gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

Giso H Dal, Alfons W Laarman, and Peter JF Lucas. Parallel probabilistic inference by weighted
model counting. In International Conference on Probabilistic Graphical Models, pp. 97–108.
PMLR, 2018.

Pim De Haan, Dinesh Jayaraman, and Sergey Levine. Causal confusion in imitation learning.

Advances in Neural Information Processing Systems, 32, 2019.

Wenhao Ding, Haohong Lin, Bo Li, and Ding Zhao. Generalizing goal-conditioned reinforcement
learning with variational causal reasoning. In Advances in Neural Information Processing Systems,
2022.

Fan Feng, Biwei Huang, Kun Zhang, and Sara Magliacane. Factored adaptation for non-stationary
reinforcement learning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho
(eds.), Advances in Neural Information Processing Systems, 2022.

Anirudh Goyal, Aniket Rajiv Didolkar, Nan Rosemary Ke, Charles Blundell, Philippe Beaudoin,
Nicolas Heess, Michael Curtis Mozer, and Yoshua Bengio. Neural production systems. In Advances
in Neural Information Processing Systems, 2021a.

Anirudh Goyal, Alex Lamb, Phanideep Gampa, Philippe Beaudoin, Charles Blundell, Sergey Levine,
Yoshua Bengio, and Michael Curtis Mozer. Factorizing declarative and procedural knowledge in
structured, dynamical environments. In International Conference on Learning Representations,
2021b.

Anirudh Goyal, Alex Lamb, Jordan Hoffmann, Shagun Sodhani, Sergey Levine, Yoshua Bengio,
and Bernhard Sch¨olkopf. Recurrent independent mechanisms. In International Conference on
Learning Representations, 2021c.

10

Under review as a conference paper at ICLR 2024

Jesse Hoey, Robert St-Aubin, Alan Hu, and Craig Boutilier. Spudd: stochastic planning using decision
diagrams. In Proceedings of the Fifteenth conference on Uncertainty in artificial intelligence, pp.
279–288, 1999.

Biwei Huang, Kun Zhang, Yizhu Lin, Bernhard Sch¨olkopf, and Clark Glymour. Generalized score
functions for causal discovery. In Proceedings of the 24th ACM SIGKDD international conference
on knowledge discovery & data mining, pp. 1551–1560, 2018.

Biwei Huang, Chaochao Lu, Liu Leqi, Jos´e Miguel Hern´andez-Lobato, Clark Glymour, Bernhard
Sch¨olkopf, and Kun Zhang. Action-sufficient state representation learning for control with
structural constraints. In International Conference on Machine Learning, pp. 9260–9279. PMLR,
2022.

Inwoo Hwang, Yunhyeok Kwak, Yeon-Ji Song, Byoung-Tak Zhang, and Sanghack Lee. On discovery
of local independence over continuous variables via neural contextual decomposition. In 2nd
Conference on Causal Learning and Reasoning, 2023.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv

preprint arXiv:1611.01144, 2016.

Łukasz Kaiser, Mohammad Babaeizadeh, Piotr Miłos, Bła˙zej Osi´nski, Roy H Campbell, Konrad
Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al. Model based
reinforcement learning for atari. In International Conference on Learning Representations, 2020.

Nan Rosemary Ke, Aniket Rajiv Didolkar, Sarthak Mittal, Anirudh Goyal, Guillaume Lajoie, Stefan
Bauer, Danilo Jimenez Rezende, Michael Curtis Mozer, Yoshua Bengio, and Christopher Pal.
Systematic evaluation of causal discovery in visual model based reinforcement learning.
In
Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track
(Round 2), 2021.

Michael Kearns and Daphne Koller. Efficient reinforcement learning in factored mdps. In IJCAI,

volume 16, pp. 740–747, 1999.

Taylor W Killian, Marzyeh Ghassemi, and Shalmali Joshi. Counterfactually guided policy transfer in
clinical settings. In Conference on Health, Inference, and Learning, pp. 5–31. PMLR, 2022.

Thomas Kipf, Elise van der Pol, and Max Welling. Contrastive learning of structured world models.

In International Conference on Learning Representations, 2020.

Sanghack Lee and Elias Bareinboim. Structural causal bandits: Where to intervene? Advances in

neural information processing systems, 31, 2018.

Yunzhu Li, Antonio Torralba, Anima Anandkumar, Dieter Fox, and Animesh Garg. Causal discovery
in physical systems from videos. Advances in Neural Information Processing Systems, 33:9180–
9192, 2020.

Chaochao Lu, Bernhard Sch¨olkopf, and Jos´e Miguel Hern´andez-Lobato. Deconfounding reinforce-

ment learning in observational settings. arXiv preprint arXiv:1812.10576, 2018.

Chaochao Lu, Biwei Huang, Ke Wang, Jos´e Miguel Hern´andez-Lobato, Kun Zhang, and Bernhard
Sch¨olkopf. Sample-efficient reinforcement learning via counterfactual-based data augmentation.
arXiv preprint arXiv:2012.09092, 2020.

Clare Lyle, Amy Zhang, Minqi Jiang, Joelle Pineau, and Yarin Gal. Resolving causal confusion in
reinforcement learning via robust exploration. In Self-Supervision for Reinforcement Learning
Workshop-ICLR, volume 2021, 2021.

Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous

relaxation of discrete random variables. arXiv preprint arXiv:1611.00712, 2016.

Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. Explainable reinforcement learning
In Proceedings of the AAAI conference on artificial intelligence, pp.

through a causal lens.
2493–2500, 2020.

11

Under review as a conference paper at ICLR 2024

Thomas Mesnard, Theophane Weber, Fabio Viola, Shantanu Thakoor, Alaa Saade, Anna Haru-
tyunyan, Will Dabney, Thomas S Stepleton, Nicolas Heess, Arthur Guez, et al. Counterfactual
credit assignment in model-free reinforcement learning. In International Conference on Machine
Learning, pp. 7654–7664. PMLR, 2021.

Suraj Nair, Yuke Zhu, Silvio Savarese, and Li Fei-Fei. Causal induction from visual observations for

goal directed tasks. arXiv preprint arXiv:1910.01751, 2019.

Michael Oberst and David Sontag. Counterfactual off-policy evaluation with gumbel-max structural
causal models. In International Conference on Machine Learning, pp. 4881–4890. PMLR, 2019.

Sherjil Ozair, Yazhe Li, Ali Razavi, Ioannis Antonoglou, Aaron Van Den Oord, and Oriol Vinyals.
Vector quantized models for planning. In International Conference on Machine Learning, pp.
8302–8313. PMLR, 2021.

Judea Pearl. Causality. Cambridge university press, 2009.

Jonas Peters, Dominik Janzing, and Bernhard Sch¨olkopf. Elements of causal inference: foundations

and learning algorithms. The MIT Press, 2017.

Silviu Pitis, Elliot Creager, and Animesh Garg. Counterfactual data augmentation using locally

factored dynamics. Advances in Neural Information Processing Systems, 33, 2020.

Silviu Pitis, Elliot Creager, Ajay Mandlekar, and Animesh Garg. MocoDA: Model-based counterfac-

tual data augmentation. In Advances in Neural Information Processing Systems, 2022.

David Poole. Context-specific approximation in probabilistic inference.

In Proceedings of the

Fourteenth conference on Uncertainty in artificial intelligence, pp. 447–454, 1998.

Joseph Ramsey, Peter Spirtes, and Jiji Zhang. Adjacency-faithfulness and conservative causal
inference. In Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence,
pp. 401–408, 2006.

Danilo J Rezende, Ivo Danihelka, George Papamakarios, Nan Rosemary Ke, Ray Jiang, Theophane
Weber, Karol Gregor, Hamza Merzic, Fabio Viola, Jane Wang, et al. Causally correct partial
models for reinforcement learning. arXiv preprint arXiv:2002.02836, 2020.

Reuven Y Rubinstein and Dirk P Kroese. The cross-entropy method: a unified approach to com-
binatorial optimization, Monte-Carlo simulation, and machine learning, volume 133. Springer,
2004.

Bernhard Sch¨olkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner,
Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. Proceedings of the
IEEE, 109(5):612–634, 2021.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon
Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari,
go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.

Maximilian Seitzer, Bernhard Sch¨olkopf, and Georg Martius. Causal influence detection for improv-
ing efficiency in reinforcement learning. In Advances in Neural Information Processing Systems,
2021.

Sumedh A Sontakke, Arash Mehrjou, Laurent Itti, and Bernhard Sch¨olkopf. Causal curiosity: Rl
agents discovering self-supervised experiments for causal representation learning. In International
conference on machine learning, pp. 9848–9858. PMLR, 2021.

Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. Causation, prediction,

and search. MIT press, 2000.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.

12

Under review as a conference paper at ICLR 2024

Yuhta Takida, Takashi Shibuya, Weihsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu Uesaka,
Naoki Murata, Shusuke Takahashi, Toshiyuki Kumakura, and Yuki Mitsufuji. Sq-vae: Variational
In International
bayes on discrete representation with self-annealed stochastic quantization.
Conference on Machine Learning, pp. 20987–21012. PMLR, 2022.

Santtu Tikka, Antti Hyttinen, and Juha Karvanen. Identifying causal effects via context-specific
independence relations. Advances in Neural Information Processing Systems, 32:2804–2814, 2019.

Manan Tomar, Amy Zhang, Roberto Calandra, Matthew E Taylor, and Joelle Pineau. Model-invariant
state abstractions for model-based reinforcement learning. arXiv preprint arXiv:2102.09850, 2021.

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in

neural information processing systems, 30, 2017.

Sergei Volodin, Nevan Wichers, and Jeremy Nixon. Resolving spurious correlations in causal models

of environments via interventions. arXiv preprint arXiv:2002.05217, 2020.

Zizhao Wang, Xuesu Xiao, Yuke Zhu, and Peter Stone. Task-independent causal state abstraction.
In Proceedings of the 35th International Conference on Neural Information Processing Systems,
Robot Learning workshop, 2021.

Zizhao Wang, Xuesu Xiao, Zifan Xu, Yuke Zhu, and Peter Stone. Causal dynamics learning for task-
independent state abstraction. In International Conference on Machine Learning, pp. 23151–23180.
PMLR, 2022.

Will Williams, Sam Ringer, Tom Ash, David MacLeod, Jamie Dougherty, and John Hughes. Hi-
erarchical quantized autoencoders. Advances in Neural Information Processing Systems, 33:
4524–4535, 2020.

Weiran Yao, Guangyi Chen, and Kun Zhang. Learning latent causal dynamics. arXiv preprint

arXiv:2202.04828, 2022.

Amy Zhang, Zachary C Lipton, Luis Pineda, Kamyar Azizzadenesheli, Anima Anandkumar, Laurent
Itti, Joelle Pineau, and Tommaso Furlanello. Learning causal state representations of partially
observable environments. arXiv preprint arXiv:1906.10437, 2019.

Amy Zhang, Clare Lyle, Shagun Sodhani, Angelos Filos, Marta Kwiatkowska, Joelle Pineau, Yarin
Gal, and Doina Precup. Invariant causal prediction for block mdps. In International Conference
on Machine Learning, pp. 11214–11224. PMLR, 2020a.

Junzhe Zhang, Daniel Kumor, and Elias Bareinboim. Causal imitation learning with unobserved

confounders. Advances in neural information processing systems, 33:12263–12274, 2020b.

Nevin L Zhang and David Poole. On the role of context-specific independence in probabilistic
inference. In 16th International Joint Conference on Artificial Intelligence, IJCAI 1999, Stockholm,
Sweden, volume 2, pp. 1288, 1999.

Artem Zholus, Yaroslav Ivchenkov, and Aleksandr Panov. Factorized world models for learning
causal relationships. In ICLR2022 Workshop on the Elements of Reasoning: Objects, Structure
and Causality, 2022.

Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Mart´ın-Mart´ın, Abhishek Joshi, Soroush Nasiriany,
and Yifeng Zhu. robosuite: A modular simulation framework and benchmark for robot learning.
arXiv preprint arXiv:2009.12293, 2020.

13

Under review as a conference paper at ICLR 2024

A APPENDIX FOR PRELIMINARY

A.1 EXTENDED RELATED WORK

Recently, incorporating causal reasoning into RL has gained much attention in the community in vari-
ous aspects. For example, causality has been shown to improve off-policy evaluation (Buesing et al.,
2019; Oberst & Sontag, 2019), goal-directed tasks (Nair et al., 2019), credit assignment (Mesnard
et al., 2021), robustness (Lyle et al., 2021; Volodin et al., 2020), policy transfer (Killian et al., 2022),
explainability (Madumal et al., 2020), and policy learning with counterfactual data augmentation (Lu
et al., 2020; Pitis et al., 2020; 2022). Causality has also been integrated with bandits (Bareinboim
et al., 2015; Lee & Bareinboim, 2018) or imitation learning (Bica et al., 2021; De Haan et al., 2019;
Zhang et al., 2020b) to handle the unobserved confounders and learn generalizable policies. Another
line of work focused on causal reasoning over the high-dimensional visual observation (Lu et al.,
2018; Feng et al., 2022; Rezende et al., 2020) where the representation learning is crucial (Zhang
et al., 2019; Sontakke et al., 2021; Tomar et al., 2021). Our work falls into the category of improving
dynamics learning by incorporating causality, where recent works have focused on the discovery of
the causal relationships between the variables explicitly (Wang et al., 2021; 2022; Ding et al., 2022).
On the contrary, our work incorporates fine-grained local causal relationships into dynamics learning,
which is underexplored in prior works.

A.2 BACKGROUND ON LOCAL INDEPENDENCE RELATIONSHIP

In this subsection, we provide the background on local independence relationship. We first describe
context-specific independence (CSI) (Boutilier et al., 2013), which denotes a variable being con-
ditionally independent of others given a particular context, not the full set of parents in the graph.

Definition 2 (Context-Specific Independence (CSI) (Boutilier et al., 2013), reproduced from Hwang
et al. (2023)). Y is said to be contextually independent of XB given the context XA = xA if
P (cid:0)y
B whenever P (xA, xB) > 0. This
xA, xB
will be denoted by Y

(cid:1), holds for all y
XA = xA.

(cid:1) = P (cid:0)y

xA
|
XB

and xB

∈ X

∈ Y

|

⊥⊥

|

CSI has been widely studied especially for discrete variables with low cardinality, e.g., binary
variables. Context-set specific independence (CSSI) generalizes the notion of CSI allowing continuous
variables.

· · ·

, Xd

Definition 3 (Context-Set Specific Independence (CSSI) (Hwang et al., 2023)). Let X =
be an
be a non-empty set of the parents of Y in a causal graph, and
X1,
{
is said to be a context set which induces context-set spe-
event with a positive probability.
(cid:1) holds for every
cific independence (CSSI) of XAc from Y if p (cid:0)y
|
(xAc, xA) , (cid:0)x′Ac , xA
. This will be denoted by Y

(cid:1) = p (cid:0)y
.
XA,

xAc , xA
XAc

x′Ac, xA

E ⊆ X

E

}

(cid:1)

|

∈ E

⊥⊥

|

E

Intuitively, it denotes that the conditional distribution p(y
different values of xAc, for all x = (xAc, xA)
variables is sufficient for modeling p(y

|
∈ E
x) when restricted in

xAc, xA) is the same for
. In other words, only a subset of the parent

x) = p(y

|

|

.

E

B APPENDIX FOR THEORETICAL ANALYSIS

B.1 PRELIMINARIES

Now, we formally define local independence by adapting CSSI to our setting. As mentioned in
Sec. 2, we consider factored MDP where the causal graph is directed bipartite. Note that X =
S1,
{
Assumption 1. We assume Markov property (Pearl, 2009), faithfulness (Peters et al., 2017), and
causal sufficiency (Spirtes et al., 2000).

, and P a(j) is parent variables of S′j.

, S′N }

, SN , A1,

S′1,
{

, Y =

, AM

· · ·

· · ·

· · ·

}

We note that these assumptions are standard in the field to properly identify the causal relationships
in MBRL (Ding et al., 2022; Wang et al., 2021; 2022; Seitzer et al., 2021; Pitis et al., 2020; 2022).

14

Under review as a conference paper at ICLR 2024

Definition 4 (Local Independence). Let T
independence S′j ⊥⊥
holds on
X
(xT c , xT ) , (cid:0)x′T c, xT
(cid:1)

T
.2

T,

E

|

\
∈ E

P a(j) and
if p(s′j |

with p(
) > 0. We say the local
E
x′T c , xT ) holds for every
xT c, xT ) = p(s′j |

E ⊆ X

⊆
E

Local independence extends conditional independence, i.e., if conditional independence S′j ⊥⊥
|
T
T holds, then local independence S′j ⊥⊥
. Local independence
holds for any
|
\
implies that only subset of the parent variables is locally relevant on
, and any other remaining
parent variables are locally irrelevant. Throughout the paper, we are concerned with the events with
the positive probability, i.e., p(

E ⊆ X

) > 0.

T,

X

E

E

\

X

T

E

Definition 5 (Local Parent Set). P a(j;
holds and S′j ̸⊥⊥
T,
P a(j;

X

T

),

E

E

\

|

) is a subset of P a(j) such that S′j ⊥⊥
for any T ⊊ P a(j;
E

).

E
E

X

\

P a(j;

)

E

|

In other words, P a(j;
Clearly, P a(j;
independence.

X

E

) is a minimal subset of P a(j) in which the local independence on

holds.
is equivalent to the (global) conditional

E

) = P a(j), i.e., local independence on

X

Definition 1 (Local Causal Graph). Local causal graph (LCG) on
E
E

P a(j;

=

(i, j)
{

)
}

∈

E

i

|

.

E ⊆ X

is

GE

= (V, E

E

) where

Local causal graph is a subgraph of the causal graph, i.e.,
relationships that arise under the event

, which describes fine-grained causal
, i.e., local independence and LCG under
are equivalent to conditional independence and CG, respectively. Analogous to the faithfulness
X
assumption (Peters et al., 2017) that no conditional independences other than ones entailed by CG are
present, we introduce a similar assumption for LCG and local independence.

GE ⊆ G
=

. Note that

GX

G

E

are present, i.e., for any j, there does not exists any T such that P a(j;
X

T

.

E

\

∅

, no local independences on

other than the ones entailed
and

T

=

)

E

|

\

E

E
T,

-Faithfulness). For any

Assumption 2 (
by
GE
S′j ⊥⊥
Regardless of the
E
However, such LCG may not be unique.
See (Hwang et al., 2023, Example. 2) for the violation of
Prop. 1.

E

E

E

-faithfulness assumption, LCG always exists because P a(j;

-faithfulness implies the uniqueness of P a(j;

) always exists.
.
) and
GE
-faithfulness. We now provide a proof of

E

E

Lemma 1 (Hwang et al. (2023), Prop. 4). S′j ⊥⊥
. Then,
Proposition 1 (Monotonicity). Let

F ⊆ E

.

GF ⊆ GE

X

P a(j;

)

E

|

P a(j;

),

E

F

\

holds for any

.

F ⊆ E

Proof. Since S′j ⊥⊥
definition; otherwise, P a(j;
) for all j and thus
P a(j;

P a(j;
)

X

\

P a(j;
)

)
|
P a(j;
.

E
\
F
GF ⊆ GE

E

),
E
=

F
∅

E

holds by Lemma 1, P a(j;
which leads to contradiction. Therefore, P a(j;

P a(j;

⊆

F

E

)

) holds by

)

F

⊆

B.2

IDENTIFIABILITY IN FACTORED MDP

Due to the nature of factored MDP where the causal graph is directed bipartite, each Markov
equivalence class constrained under temporal precedence contains a single unique causal graph (i.e., a
skeleton determines a unique causal graph since temporal precedence fully orients the edges). Given
this background, it is known that the causal graph is uniquely identifiable with oracle conditional
independence test (Ding et al., 2022) or score-based method (Brouillard et al., 2020). Similarly, we
now show that LCG is also identifiable via score maximization.

2T denotes an index set of T.

15

̸
̸
Under review as a conference paper at ICLR 2024

To begin with, we recall the score function in Eq. (4):

z,

K
z=1)
(
z
S
}
{G
E
:= sup E (cid:2)log ˆp(s′
Ep(s,a,s′)
:= sup
ϕ

Ep(s,a)Ep(s′

s,a)
|

= sup

ϕ

= sup

(cid:90)

(cid:88)

s, a;
(cid:2)log ˆp(s′

|

{G

)

}

z,

z
E
s, a;

−
z,

{G

|
(cid:2)log ˆp(s′

λ

(cid:3)
|
, ϕ)
}

z
|G
z
E

λ

z
|G

−

(cid:3) ,
|

s, a;

z,

{G

, ϕ)
}

z
E

λ

z
|G

−

|

(cid:3) ,
|

p(s, a)

(cid:16)

Ep(s′

s,a) log ˆp(s′
|

|

s, a;

z, ϕ)

λ

z
|G

−

G

(cid:17)
|

,

ϕ

z

(cid:88)

= sup

ϕ

z

where

z

∈E

(s,a)
(cid:34)(cid:90)

(s,a)

∈E

z

p(s, a)

(cid:16)

Ep(s′

s,a) log ˆp(s′
|

|

s, a;

G

(cid:17)

z, ϕ)

λ

·

−

p(

z)

E

ˆp(s′

|

s, a;

z, ϕ) =

G

(cid:89)

j

ˆpj(s′j |

P a(j, z), z; ϕj),

(8)

(9)

(10)

(11)

(12)

(13)

(cid:35)

,

· |G

z

|

ϕ1,
{

ϕ :=
, ϕN
function ˆpj. Specifically, for all z
z. We denote ˆp
(s, a)

, and each ϕj is a neural network which outputs the parameters of the density
z) as an input if
[K], ϕj takes P a(j, z) (i.e., parents of s′j in
z, ϕ).

∈
,ϕ := ˆp(s′

G
s, a;

s, a;

· · ·

z,

}

∈ E

, ϕ) and ˆp
}
G
Assumption 3 (Sufficient capacity). The ground truth density p(s′
decomposition

with corresponding true LCGs

, where

z
E

{G

z
E

{G

z,

}

|

|

z,ϕ := ˆp(s′
s, a)

|
(

G
∗z ,

z
E

∈ H

{G

z

{E

}

∗z ,

(
{G

) :=
}

z
E

p
{

H

| ∃

∗
z ,

.

z

,ϕ}

}

E

{G

{G

∗z }
ϕ s.t. p = ˆp

) for any
}

(14)

In other words, the model has sufficient capacity to represent the ground truth density. We assume the
density ˆp

,ϕ is strictly positive for all ϕ.

∗
z ,

z

E

}

{G

Assumption 4 (Finite differential entropy).

Lemma 2 (Finite score). Let

∗z be a true LCG on
G

|

Ep(s,a,s′) log p(s′

s, a)

<

|

|
z for all z. Then,
E

|S

.

∞
(
{G

∗z ,

K
z=1)
}
|

z

E

<

.

∞

Proof. First,

0

DKL(p

ˆp

≤
∥
E
= Ep(s,a,s′) log p(s′

{G

z

}

∗
z ,

,ϕ)
s, a)

|

−

where the equality holds because Ep(s,a,s′) log p(s′

|

s, a) <

Ep(s,a,s′) log ˆp(s′

sup
ϕ

s, a;

∗z ,

{G

, ϕ)
}

z
E

≤

|

∞
Ep(s,a,s′) log p(s′

s, a).

|

Ep(s,a,s′) log ˆp(s′

s, a;

∗z ,

z
E

, ϕ),

{G

|
}
by Assumption 4. Therefore,

On the other hand, by Assumption 3, there exists ϕ∗ such that p = ˆp

,ϕ∗ . Hence,

Ep(s,a,s′) log ˆp(s′

sup
ϕ

s, a;

∗z ,

{G

, ϕ)
}

z
E

|

{G
Ep(s,a,s′) log ˆp(s′

≥
= Ep(s,a,s′) log p(s′

z

E
}
s, a;

∗
z ,

|

∗z ,

{G

, ϕ∗)
}

z
E

s, a).

|
s, a). Therefore,
(cid:3) .

|
E (cid:2)

∗z |

|G

|

|

by Assumption 4, this concludes that

∞

(cid:88)

z

p(

z)(

E

z
|G

| − |G

).
∗z |

(21)

Thus, supϕ

Ep(s,a,s′) log ˆp(s′

s, a;

, ϕ) = Ep(s,a,s′) log p(s′

∗z ,

z
E

|

{G

}
z=1) = Ep(s,a,s′) log p(s′
K
}
Ep(s,a,s′) log p(s′

s, a)

|
<

s, a)

λ

·

−

(

∗z ,

{G

Since
(

z
S
E
N (N + M ) and
∗z | ≤
K
z=1)
z
|
|S
}
E
Lemma 3 (Score difference). Let

|G
∗z ,

{G

∞

<

|

.

∗z be a true LCG on
G

z for all z. Then,
E

(

S

∗z ,

{G

K
z=1)
}

z
E

,

(
{G

z

E

K
z=1) = inf
}
ϕ

− S

DKL(p

∥

16

ˆp

z,

,ϕ) + λ

z

}

E

{G

(15)

(16)

(17)

(18)

(19)

(20)

Under review as a conference paper at ICLR 2024

Proof. First, we can rewrite the score

z,

(
{G

S

z
E

}

K
z=1) = sup
ϕ

z,
(
{G
Ep(s,a,s′) log ˆp(s′

S

K
z=1) as:
z
E
}
s, a;

z,

|

{G

z

, ϕ)
}

−

λ

·

E

=

=

−

−

inf
ϕ −
inf
ϕ

Ep(s,a,s′) log ˆp(s′

DKL(p

ˆp

{G

z,

z
E

}

∥

s, a;

z,

, ϕ)
}

z
E

{G

−
|
,ϕ) + Ep(s,a,s′) log p(s′

λ

(22)

(23)

·

E (cid:2)

(cid:3)
|
and thus the score

z
|G

(24)

E (cid:2)

(cid:3)

z
|G
|
E (cid:2)

·
s, a)

(cid:3)

|
λ

z
|G

−

|
K
z=1)
|
}

z

The last equality holds by Assumption 4. By Lemma 2,
difference

K
z=1)

(

∗z ,
E
K
z=1) is well defined. Using Eq. (20), we obtain:
z,
{G
}
K
z=1) = inf
z
}
E
ϕ

,ϕ) + λ

DKL(p

(
{G

z
|G

(cid:88)

z)(

z
E

∞

p(

|S

<

z
E

{G

ˆp

E

∥

z,

}

z

− S
z,

S
∗z ,

∗z ,
(
z
E
{G
K
z=1)
z
}
E

}

(
{G

− S

(

S

{G

| − |G

).
∗z |

= inf
ϕ

= inf
ϕ

= inf
ϕ

= inf
ϕ

(cid:88)

=

(cid:90)

(cid:88)

z
(cid:88)

z
(cid:88)

z

(s,a)

∈E
(cid:90)

z

p(

z)

E

z
{E

Proposition 2. Let
z that maximizes the score:
E
each ˆ
G
ˆ
=
G
G

K
z=1 ∈
}
z is true LCG on corresponding
where

ˆ
z
G
{

G

K
K
z=1 be the graphs on each
z=1 be the arbitrary decomposition. Let
}
}
K
z=1). With the Assumptions 1 to 4,
argmax
}
z for small enough λ > 0. In particular, if K = 1, then
E

(
{G

z
E

} S

z
{G

z,

{

ˆ
z
G

is the ground truth causal graph.

It is enough to show that

∗z ,

(
{G

S

z
E

}

K
z=1) >

∗z be a true LCG on
=

E

S

(
{G

Proof. Let
z,
z
(
z
{G
}
E
DKL(p

G
K
z=1) if
∗z ̸
G
}
K
z=1)
∗z ,
− S
z,

S
= inf
ϕ

ˆp

{G

∥

z
E

}

z,

K
z=1)
}
{G
(cid:88)
,ϕ) + λ

z
E

p(

z for some z. By Lemma 3,
G
(

z for all z.
E

z)(

z
|G

| − |G

)
∗z |

E

z

(cid:90)

p(s, a)DKL(p(

· |

s, a)

∥

ˆp

z,

{G

E

,ϕ(

z

}

· |

s, a)) + λ

(cid:88)

z

p(

z)(

E

z
|G

| − |G

)

∗z |

p(s, a)DKL(p(

· |

s, a)

ˆp(

· |

∥

s, a;

z, ϕ)) + λ

G

z)(

E

z
|G

| − |G

)
∗z |

pz(s, a)DKL(p(

s, a)

ˆp(

· |

∥

s, a;

· |

z, ϕ)) + λ

G

p(

z)(

E

z
|G

| − |G

)

∗z |

(cid:88)

p(

z
(cid:88)

z

p(

z)DKL(pz

E

z,ϕ) + λ

ˆp
G

∥

p(

z)(

E

z
|G

| − |G

)

∗z |

(cid:88)

z
(cid:88)

)
∗z |

p(

z)(

z
|G

p(

z) inf
ϕ

DKL(pz

z,ϕ) + λ
ˆp
G

z

z

.

∥

∥

(cid:20)

E

E

E

=

z)

p(

z
|G

inf
ϕ

| − |G

z
(cid:88)

DKL(pz

z,ϕ) + λ(
ˆp
G

| − |G
(cid:21)
)
∗z |
z), i.e., density function of the distribution PS
Here, pz(s, a) = p(s, a)/p(
E
z. For brevity, we denote DKL(pz
action variables restricted to
E
s, a)
z,ϕ) + λ(
s, a;
∥
all z
∈
Case 0:
Case 1:
Case 2:
inf ϕ DKL(pz
and (ii)

· |
∗z =
G
G
⊊
∗z
|G
G
G
z. In this case, there exists (i
j)
∗z ̸⊆ G
G
z,ϕ) > 0. We consider two subcases: (i)
ˆp
∥
G−z :=
G
′,
z
G
∈
G−z . Then,

G
z. Clearly, Az = 0 in this case.
z. Then,
>

∗z ̸
G
and thus Az > 0 since λ > 0.

ˆp(
[K], Az > 0 if and only if

z, ϕ)) and Az := inf ϕ DKL(pz

∈ G
z
∈
G
. Clearly, if

ˆp
∥
G
z
| − |G
|G

∗z ̸⊆ G

∗z |}

z
|G

ˆp
G

G+

| G

{G

∗z |

∗z |

→

|G

|G

<

=

z.

G

∥

|

|

′

′

z
G

∈

A

z , i.e., state and

×

|E

z,ϕ) := (cid:82) pz(s, a)DKL(p(

· |
). We will show that for

z and thus
∗z such that (i
j) /
∈ G
→
z :=
,
′,
∗z |}
∗z ̸⊆ G
′
′
|G
| G
{G
G+
z then Az > 0. Suppose
z
G

| ≥ |G

∈

(25)

(26)

(27)

(28)

(29)

(30)

(31)

(32)

ηz :=

λ

≤

1
N (N + M ) + 1
′,ϕ)

λ

≤
inf
ϕ

inf ϕ DKL(pz

∥

ˆp
G
N (N + M ) + 1
ˆp
G

DKL(pz

′,ϕ) + λ(

∥

G

<

z
|G

min
G−
z

′

DKL(pz

inf
ϕ

∈
inf ϕ DKL(pz

∥
ˆp
G

′,ϕ)

ˆp
G

′,ϕ)

for

′

∀G

∈

G−z

∥
|G∗z | − |G′|
for
) > 0
∗z |

G−z .

′

∀G

∈

| − |G

17

⇐⇒

⇐⇒

(33)

(34)

(35)

Under review as a conference paper at ICLR 2024

Here, we use the fact that
for 0 <
λ
∀
≤
K
z=1) > 0 if
z,
z
S
}
E
ˆp
inf ϕ DKL(pz
G

(
{G

∥

∗z | − |G
∗z |
ηz. Consequently, for 0 < λ
=

| ≤ |G

|G

′

< N (N + M ) + 1. Therefore, Az > 0 if
G
) := minz ηz, we have
−
}
z for some z. We note that (i) ηz > 0 since G−z is finite and
}
G

S
) = minz ηz > 0.

G−z , and thus (ii) η(

=
∗z ̸
G
K
z=1)
z
}
E

z
{E

{G

{E

η(

∗z ,

≤

∈

(

z

z

′

∗z ̸
G
G
′,ϕ) > 0 for any

η(

}∈T

z
{E

z
{E

Assumption 5. inf

is a set of all decompositions of size K.

) > 0 where
}
. With
Recall that Prop. 2 holds for 0 < λ
z
{E
}
≤
{E
}
), which allows Prop. 2 to hold on any arbitrary
Assumption 5, we now let 0 < λ
inf
z
{E
{E
}∈T
decomposition. It is worth noting that
is a highly complex mathematical object, which makes
T
it challenging to prove or find a counterexample of the above assumption. In general, for a small
K
fixed λ > 0, the arguments henceforth would hold for decompositions
z=1 |
η(

) > 0 for any decomposition
}

) and η(
η(

} ∈ T

z
{E

as λ

λ =

{{E

{E

η(

≤

0.

T

}

}

z

z

z

z

)

}

, where
λ
}

z
{E

λ
→ T
T
≥
z, ˆ
ˆ
K
Proposition 3. Let
z
z=1 ∈
{
}
E
G
on ˆ
[K]. Also, let
z for all z
∈
E
corresponding true LCGs on each
for small enough λ > 0.

→
argmax

z

(

z,

{G

z,

z=1). Then, each ˆ
K
z
G
}
E
K
z=1 be the arbitrary decomposition and
}
(cid:3)

{G
{F
z. Then, with the Assumptions 1 to 5, E(cid:2)
ˆ
z
|
G
F

} S

E

|

z

z is the true LCG
K
z=1 be the
(cid:3) holds

z
}
{G
E (cid:2)
z
|G
≤

|

η(

}∈T

inf

z
{E

z
{E

).
}

First,

K
Proof. Let 0 < λ
z=1 ∈
≤
}
ˆ
K
z=1 also maximizes the score on the fixed
implies that
z
}
G
{
z, ˆ
argmax
(
z
z
} S
E
{G
{G
K
z=1 is the arbitrary decomposition,
since
z
}
z, by Eq. (20),
true LCGs on each
z=1) = Ep(s,a,s′) log p(s′
K

z is true LCG on
)
(

z=1). Thus, ˆ
K
G

z, ˆ
ˆ
z
E
G
{

z, ˆ
ˆ
z
E
G
{

(
{G

s, a)

{F

z,

S

λ

}

}

z

(cid:88)

(

K
argmax
z,
z=1)
z
z,
{G
E
}
{G
ˆ
ˆ
K
K
z=1,
z
z
z=1 ∈
G
{
}
E
{
}
[K] by Prop. 2. Also,
z for all z
∈
E
is the
) holds. Since
}
≥ S

z
} S
E
i.e.,

z
{G

z,

F

}

z

F
(
{G

S

F

}

|

−

Similarly,

(

z, ˆ
ˆ
z
E
G
{

}

S

z=1) = Ep(s,a,s′) log p(s′
K

Therefore, 0

(

z, ˆ
ˆ
z
E
G
{

)
}

(

z,

{G

− S

≤ S

) = E (cid:2)
}

z
|G

z
F

s, a)

E

−

|

(cid:3)
|

p(

F

z)

z

.

|

· |G

p( ˆ
E

z)

ˆ
z
G

.
|

· |

(36)

(37)

z

(cid:88)

λ

−
(cid:104)
|

z
(cid:105)
|

ˆ
z
G

holds.

We note that Prop. 3 can be further generalized because a partition of size K can express any partition
J
K
k=1 can express the partition
of size J
j=1 by letting
}
be a partition of
1,
1 =

j
{D

J .

D

≤
· · ·

1, and
J
E
E
Theorem 1 (Identifiability). Let
. Suppose

K. For example, the partition
k
}
{E
,
,
K
E
}
K
z=1 ∈
}
GD

J ,
· · ·
{E
z, ˆ
ˆ
z
{
E
G
=
GF

where

J
D

1 =

⊊

⊊

G

−

−

X

GD

D
with the Assumptions 1 to 5, there exists I ⊊ [K] such that (i) (cid:83)
(ii) ˆ
G

for all z /
∈

I and ˆ
G

for all z

z =

z =

GD

I.

∈

G

I

i

∈

for any

argmax

z,
z
} S
E
, and

{G
F ⊆ D

D
(
{G
GF
ˆ
i
E

z,
=

z
E
G
⊆ D

K
z=1) and K > 1. Let
}
c. Then,
for any
ˆ
i) = 0,
and p(
E

F ⊆ D
(cid:83)
I
i
∈

D \

Proof. For brevity, we denote the conditions
for any
z
some J ⊊ [K]. Let
thus
z =
GD
(ii) Therefore,

c as condition (ii). Let
K
z=1 be the LCGs corresponding to each
}

by Prop. 1 and condition (i). For any z /
∈

F ⊆ D

for any

z
{G

{F

GD

J,

=

F

G

z

z=1 be the decomposition such that (cid:83)
GF
K
}

F ⊆ D

z. For any z
F
c and thus
⊆ D

GF

GF
j =

G
for
J F
D
and
z
F
⊆ D
by condition

j
∈
J,

G

∈
=

as condition (i) , and

=

holds. Similarly, we let I
and condition (i). Also, for all z /
∈
condition (ii). Therefore,
G
z /
∈

GT
I. Combining together,
(cid:105)

z =

∈

(cid:104)

E

ˆ
z
G
|

|

(cid:88)

=

z

E (cid:2)

(cid:3) =
|

z
|G

(cid:88)

p(

z)

z
|G

|

= p(

D

F

z
[K] such that I =
I, let
T
. Since

ˆ
z
z
|
{
E
z := ˆ
z
E
\ D ̸
ˆ
z,
z
GT
E

⊆

T

⊆ D}
=
∅
z ⊆

c)

|G|

+ p(

)
|GD|
D
. Then, ˆ
for all z
∈
G
GD
c and thus
. Then,
z
GT
T
⊆ D
z by Prop. 1. Therefore, ˆ
ˆ
G
G

z =

(38)

I by Prop. 1
c by
z =
GD
for all
z =
G

p( ˆ
E

z)
|

ˆ
z
G

|

= p(
i

(cid:91)

I

∈

ˆ
i)
E

|GD|

(cid:91)

+ p(

I

i /
∈

ˆ
i)
E

|G|

(39)

18

Under review as a conference paper at ICLR 2024

Figure 6: Illustration of CHEMICAL (full-fork) environment with 4 nodes. (Left) the color of the root
node determines the activation of local causal graph fork. (Right) the noisy nodes are redundant for
predicting the colors of other nodes under the local causal graph.

holds. Recall that E
z
|G
subtracting Eq. (39) from Eq. (38),

ˆ
z
G

≤

|

E (cid:2)

(cid:104)
|

(cid:105)

(cid:3) holds by Prop. 3. Also, by definition of I, (cid:83)
|

i

I

∈

ˆ
i
E

. By

⊆ D

p(

D \

ˆ
i)
E

(cid:91)

i

I

∈

(

)
|GD| − |G|

·

≥

0

(40)

(cid:83)

ˆ
i) = 0.
E

i

I

∈

D \

holds. Since

<

|G|

|GD|

, this is only possible when p(

C APPENDIX FOR EXPERIMENTS

C.1 EXPERIMENTAL DETAILS

C.1.1 PLANNING ALGORITHM

To assess the performance of different dynamics models of the baselines and our method, we use
a model predictive control (MPC) (Camacho & Alba, 2013) which selects the actions based on the
prediction of the learned dynamics model, following prior works (Ding et al., 2022; Wang et al.,
2022). Specifically, we use a cross-entropy method (CEM) (Rubinstein & Kroese, 2004), which
iteratively generates and refines action sequences through a process of sampling from a probability
distribution that is updated based on the performance of these sampled sequences, with a known
reward function. We use a random policy for the initial data collection.

Table 4: Environment configurations.

Table 5: CEM parameters.

Parameters

full-fork

full-chain

Chemical

Training step
Optimizer
Learning rate
Batch size
Initial step
Max episode length
Action type

1.5

105

×
Adam
1e-4
256
1000
25
Discrete

1.5

105

×
Adam
1e-4
256
1000
25
Discrete

Magnetic

2

105
×
Adam
1e-4
256
2000
25
Continuous

Chemical

Magnetic

CEM parameters

full-fork

full-chain

Planning length
Number of candidates
Number of top candidates
Number of iterations
Exploration noise
Exploration probability

3
64
32
5
N/A
0.05

3
64
32
5
N/A
0.05

1
64
32
5
1e-4
N/A

C.1.2 CHEMICAL

Here, we describe two settings, namely full-fork and full-chain, modified from Ke et al. (2021).
In both settings, there are 10 state variables representing the color of corresponding nodes, with
each color represented as a one-hot encoding. The action variable is a 50-dimensional categorical
variable that changes the color of a specific node to a new color (e.g., changing the color of the third
node to blue). According to the underlying causal graph and pre-defined conditional probability
distributions, implemented with randomly initialized neural networks, an action changes the colors of

19

{},,{},Root nodeFullForkRoot node:  Noisy node?: Color},,,,{: Action????Under review as a conference paper at ICLR 2024

(a)

(b)

(c)

Figure 7: (a) Causal graph of Magnetic environment. Red boxes indicate redundant edges under
the non-magnetic context. (b) LCG under the magnetic context. (c) LCG under the non-magnetic
context.

the intervened object’s descendants as depicted in Fig. 6. As shown in Fig. 3(a), the (global) causal
graph is full in both settings, and the LCG is fork and chain, respectively. For example in full-fork,
the LCG fork is activated according to the particular color of the root node, as shown in Fig. 6.

In both settings, the task is to match the colors of each node to the given target. The reward function
is defined as:

r =

1
O

|

(cid:88)

i

|

O

∈

1 [si = gi] ,

(41)

where O is a set of the indices of observable nodes, si is the current color of the i-th node, and gi is
the target color of the i-th node in this episode. Success is determined if all colors of observable nodes
are the same as the target. During training, all 10 nodes are observable, i.e., O =
. In
downstream tasks, the root color is set to induce the LCG, and the agent receives noisy observations
for a subset of nodes, aiming to match the colors of the rest of the observable nodes. As shown in
Fig. 6, noisy nodes are spurious for predicting the colors of other nodes under the LCG. Thus, the
agent capable of reasoning the fine-grained causal relationships would generalize well in downstream
(0, σ2), similar to Wang et al.
tasks.3 To create noisy observations, we use a noise sampled from
(2022). Specifically, the noise is multiplied to the one-hot encoding representing color during the test.
In our experiments, we use σ = 100.

, 9
}

0,
{

· · ·

N

As the root color determines the local causal graph in both settings, the root node is always observable
to the agent during the test. The root colors of the initial state and the goal state are the same, inducing
the local causal graph. As the root color can be changed by the action during the test, this may pose a
challenge in evaluating the agent’s reasoning of local causal relationships. This can be addressed by
modifying the initial distribution of CEM to exclude the action on the root node and only act on the
other nodes during the test. Nevertheless, we observe that restricting the action on the root during the
test has little impact on the behavior of any model, and we find that this is because the agent rarely
changes the root color as it already matches the goal color in the initial state.

C.1.3 MAGNETIC

In this environment, there are two objects on a table, a moving ball and a box, colored either red or
black, as shown in Fig. 3(b). The red color indicates that the object is magnetic. In other words, when
they are both colored red, magnetic force will be applied and the ball will move toward the box. If
one of the objects is colored black, the ball would not move since the box has no influence on the ball.
The state consists of the color, x, y position of each object, and x, y, z position of the end-effector
of the robot arm, where the color is given as the 2-dimensional one-hot encoding. The action is a
3-dimensional vector that moves the robot arm. The causal graph of the Magnetic environment is
shown in Fig. 7(a). LCGs under magnetic and non-magnetic event are shown in Figs. 7(b) and 7(c),
respectively. The table in our setup has a width of 0.9 and a length of 0.6, with the y-axis defined by
the width and the x-axis defined by the length. For each episode, the initial positions of a moving ball
and a box are randomly sampled within the range of the table.

3We note that the transition dynamics of the environment is the same in training and downstream tasks.

20

Under review as a conference paper at ICLR 2024

The task is to move the robot arm to reach the moving ball. Thus, accurately predicting the trajectory
of the ball is crucial. The reward function is defined as:

r = 1

tanh(5

eef

−

· ∥

g

1),
∥

−

(42)

∈

R3, and (bx, by)
R3 is the current position of the end-effector, g = (bx, by, 0.8)
where the eef
is the current position of the moving ball. Success is determined if the distance is smaller than 0.05.
During the test, the color of one of the objects is black and the box is located at the position unseen
(0, σ2) during the test. Note
during the training. Specifically, the box position is sampled from
that the box can be located outside of the table, which never happens during the training. In our
experiments, we use σ = 100.

N

∈

C.2

IMPLEMENTATION DETAILS

For all methods, the dynamics model outputs the parameters of categorical distribution for discrete
variables, and the mean and standard deviation of normal distribution for continuous variables. Each
method has a similar number of model parameters. All experiments were processed using a single
GPU (NVIDIA RTX 3090). Environmental configurations and CEM parameters are shown in Table 4
and Table 5, respectively. Detailed parameters of each model are shown in Table 6.

MLP and Modular. MLP models the transition dynamics as p(s′
network for each state variable, i.e., (cid:81)

s, a). Modular has a separate
s, a), where each network is implemented as an MLP.

|

j p(s′j |

∈ {

4, 15, 20

GNN, NPS, and CDL. We employ publicly available source codes.4 For NPS, we search the number
. CDL infers the causal structure utilizing conditional mutual information
of rules N
and models the dynamics as (cid:81)
}
P a(j)). For CDL, we search the initial conditional mutual
information (CMI) threshold ϵ
and exponential moving average
. As CDL is a two-stage method, we only report their
(EMA) coefficient τ
}
final performance.

j p(s′j |
∈ {
0.9, 0.95, 0.99, 0.999

0.001, 0.002, 0.005, 0.01, 0.02

∈ {

}

GRADER. We implement GRADER based on the code provided by the authors.5 GRADER relies
on explicit conditional independence testing to discover the causal structure. In Chemical, we ran the
conditional independence test for every 10 episodes, following their default setting. We only report
their performance in Chemical due to its poor scalability on the conditional independence test in
Magnetic environment, which took about 30 minutes for each test.

Oracle and NCD. For a fair comparison, we employ the same architecture for the dynamic models
of Oracle, NCD, and our method, as their main difference lies in the inference of local causal graphs
(LCG). As illustrated in Fig. 8, the key difference is that NCD performs direct inference of the LCG
from each individual sample (referred to as sample-specific inference), while our method decomposes
the data domain and infers the LCGs for each event (referred to as event-specific inference). We
provide an implementation details of our method in the next subsection.

C.2.1

IMPLEMENTATION DETAILS OF OUR METHOD.

We use MLPs for the implementation of genc, gdec, and ˆp, with configurations provided in Table 6.
The quantization encoder genc of our method or the auxiliary network of NCD shares the initial
feature extraction layer with the dynamics model ˆp as we found that it yields better performance
compared to full decoupling of them.
Masked dynamics model. For the implementation of ˆp(s′j |
s, a; Aj), we simply mask out the
features of unused variables, but other design choices such as Gated Recurrent Unit (Chung et al.,
2014; Ding et al., 2022) are also possible. As architectural design is not the primary focus of
our work, we leave the exploration of different architectures to future work. Recall Eq. (3) that
z, the dynamics prediction model takes not only
p(s′j |
s, a) = p(s′j |
z), but also z as an input. This is because the transition function could be different among
P a(j;
E
partitions with the same LCG in general. Here, z guides the network to learn (possibly) different
transition functions even if the LCG is the same. Recall that each latent code ez

z), z) for (s, a)
E

P a(j;

C =

∈ E

ez

∈

K
z=1
}

{

4https://github.com/wangzizhao/CausalDynamicsLearning
5https://github.com/GilgameshD/GRADER

21

Under review as a conference paper at ICLR 2024

Figure 8: Comparison of the sample-specific inference of NCD (Hwang et al., 2023) (top) and
event-specific inference of our method (bottom).

denotes the partition, ˆp takes a one-hot encoding of size K according to the latent code as the
additional input to deal with such cases.

Backpropagation. We now describe how each component of our method are updated by the training
objective

total =

pred +

quant.

L

L

L

• In Eq. (6),

pred updates the encoder genc(s, a), decoder gdec(e), and the dynamics model ˆp.
L
Recall that A
pred updates the quantization decoder
gdecthrough e. During the backward path in Eq. (5), we copy gradients from e (= input of gdec)
to h (= output of genc), following a popular trick used in VQ-VAE (Van Den Oord et al., 2017).
By doing so,

pred also updates the quantization encoder genc and h.

gdec(e), backpropagation from A in

∼

L

• In Eq. (7),

L

of the codebook C since h is updated with

L
quant updates genc and the codebook C. We note that

pred.

L

pred also affects the learning

L

Hyperparameters. For all experiments, we fix the codebook size K = 16, regularization coefficient
λ = 0.001, and commitment coefficient β = 0.25, as we found that the performance did not vary
and β
much for any K > 2, λ

0.1, 0.25

3, 10−

4, 10−

10−

2

∈ {

}

∈ {

.
}

C.3 ADDITIONAL EXPERIMENTAL RESULTS AND DISCUSSIONS

C.3.1 DETAILED ANALYSIS OF LEARNED LCGS

LCGs learned by our method with a codebook size of 4 in Chemical are shown in Figs. 9 and 10.
Among the 4 codes, one (Fig. 9(b)) or two (Fig. 10(b)) represent the local causal structure fork. Our
method successfully infers the proper code for most of the OOD samples (Figs. 9(c) and 10(c)). Two
sample runs of our method with a codebook size of 4 in Magnetic are shown in Figs. 11 and 12.
Our method successfully learns LCGs correspond to a non-magnetic event (Figs. 11(d), 11(g), 12(d)
and 12(f)) and magnetic event (Figs. 11(e), 11(f), 12(e) and 12(g)).

We also observe that our method discovers more fine-grained events. Recall that the non-magnetic
event is determined when one of the objects is black, the box would have no influence on the ball
regardless of the color of the box when the ball is black, and vice versa. As shown in Fig. 13, our
method discovers the event where the ball is black (Fig. 13(b)), and the event where the box is black
(Fig. 13(a)).

We observe that the training of latent codebook with vector quantization is often unstable when
K = 2. We demonstrate the success (Fig. 14) and failure (Fig. 15) cases of our method with a
codebook size of 2. In a failure case, we observe that the embeddings frequently fluctuate between

22

EncoderDecoder(a) Local Causal Graph Inference(b) Masked Prediction with LCGState or Action VariablesMasked VariablesDynamics Model<latexit sha1_base64="7o47/186CuYTcMwNodu2vFO8Xng=">AAAB7XicbVDLSgNBEOyNrxhfUY9eBoMQQZZd8XUMePEYwTwgWcLsZDYZMzuzzMwKYck/ePGgiFf/x5t/4yTZgyYWNBRV3XR3hQln2njet1NYWV1b3yhulra2d3b3yvsHTS1TRWiDSC5VO8SaciZowzDDaTtRFMchp61wdDv1W09UaSbFgxknNIjxQLCIEWys1KzqM4RPe+WK53ozoGXi56QCOeq98le3L0kaU2EIx1p3fC8xQYaVYYTTSambappgMsID2rFU4JjqIJtdO0EnVumjSCpbwqCZ+nsiw7HW4zi0nTE2Q73oTcX/vE5qopsgYyJJDRVkvihKOTISTV9HfaYoMXxsCSaK2VsRGWKFibEBlWwI/uLLy6R57vpX7uX9RaXm5nEU4QiOoQo+XEMN7qAODSDwCM/wCm+OdF6cd+dj3lpw8plD+APn8wcZR44d</latexit>(s,a)<latexit sha1_base64="rLydIPTJEtYRfoZpDbg2DKGKs/w=">AAAB6XicbVDLSgNBEOyNrxhfUY9eBoPoKeyKr2PAi8co5gHJEmYnvcmQ2dllZlYIS/7AiwdFvPpH3vwbJ8keNLGgoajqprsrSATXxnW/ncLK6tr6RnGztLW9s7tX3j9o6jhVDBssFrFqB1Sj4BIbhhuB7UQhjQKBrWB0O/VbT6g0j+WjGSfoR3QgecgZNVZ60Ke9csWtujOQZeLlpAI56r3yV7cfszRCaZigWnc8NzF+RpXhTOCk1E01JpSN6AA7lkoaofaz2aUTcmKVPgljZUsaMlN/T2Q00nocBbYzomaoF72p+J/XSU1442dcJqlByeaLwlQQE5Pp26TPFTIjxpZQpri9lbAhVZQZG07JhuAtvrxMmudV76p6eX9RqVXzOIpwBMdwBh5cQw3uoA4NYBDCM7zCmzNyXpx352PeWnDymUP4A+fzBzxjjR4=</latexit>s0<latexit sha1_base64="7o47/186CuYTcMwNodu2vFO8Xng=">AAAB7XicbVDLSgNBEOyNrxhfUY9eBoMQQZZd8XUMePEYwTwgWcLsZDYZMzuzzMwKYck/ePGgiFf/x5t/4yTZgyYWNBRV3XR3hQln2njet1NYWV1b3yhulra2d3b3yvsHTS1TRWiDSC5VO8SaciZowzDDaTtRFMchp61wdDv1W09UaSbFgxknNIjxQLCIEWys1KzqM4RPe+WK53ozoGXi56QCOeq98le3L0kaU2EIx1p3fC8xQYaVYYTTSambappgMsID2rFU4JjqIJtdO0EnVumjSCpbwqCZ+nsiw7HW4zi0nTE2Q73oTcX/vE5qopsgYyJJDRVkvihKOTISTV9HfaYoMXxsCSaK2VsRGWKFibEBlWwI/uLLy6R57vpX7uX9RaXm5nEU4QiOoQo+XEMN7qAODSDwCM/wCm+OdF6cd+dj3lpw8plD+APn8wcZR44d</latexit>(s,a)CodebookEvent-Speciﬁc LCGsQuantizationDynamics Model<latexit sha1_base64="7o47/186CuYTcMwNodu2vFO8Xng=">AAAB7XicbVDLSgNBEOyNrxhfUY9eBoMQQZZd8XUMePEYwTwgWcLsZDYZMzuzzMwKYck/ePGgiFf/x5t/4yTZgyYWNBRV3XR3hQln2njet1NYWV1b3yhulra2d3b3yvsHTS1TRWiDSC5VO8SaciZowzDDaTtRFMchp61wdDv1W09UaSbFgxknNIjxQLCIEWys1KzqM4RPe+WK53ozoGXi56QCOeq98le3L0kaU2EIx1p3fC8xQYaVYYTTSambappgMsID2rFU4JjqIJtdO0EnVumjSCpbwqCZ+nsiw7HW4zi0nTE2Q73oTcX/vE5qopsgYyJJDRVkvihKOTISTV9HfaYoMXxsCSaK2VsRGWKFibEBlWwI/uLLy6R57vpX7uX9RaXm5nEU4QiOoQo+XEMN7qAODSDwCM/wCm+OdF6cd+dj3lpw8plD+APn8wcZR44d</latexit>(s,a)<latexit sha1_base64="rLydIPTJEtYRfoZpDbg2DKGKs/w=">AAAB6XicbVDLSgNBEOyNrxhfUY9eBoPoKeyKr2PAi8co5gHJEmYnvcmQ2dllZlYIS/7AiwdFvPpH3vwbJ8keNLGgoajqprsrSATXxnW/ncLK6tr6RnGztLW9s7tX3j9o6jhVDBssFrFqB1Sj4BIbhhuB7UQhjQKBrWB0O/VbT6g0j+WjGSfoR3QgecgZNVZ60Ke9csWtujOQZeLlpAI56r3yV7cfszRCaZigWnc8NzF+RpXhTOCk1E01JpSN6AA7lkoaofaz2aUTcmKVPgljZUsaMlN/T2Q00nocBbYzomaoF72p+J/XSU1442dcJqlByeaLwlQQE5Pp26TPFTIjxpZQpri9lbAhVZQZG07JhuAtvrxMmudV76p6eX9RqVXzOIpwBMdwBh5cQw3uoA4NYBDCM7zCmzNyXpx352PeWnDymUP4A+fzBzxjjR4=</latexit>s0<latexit sha1_base64="7o47/186CuYTcMwNodu2vFO8Xng=">AAAB7XicbVDLSgNBEOyNrxhfUY9eBoMQQZZd8XUMePEYwTwgWcLsZDYZMzuzzMwKYck/ePGgiFf/x5t/4yTZgyYWNBRV3XR3hQln2njet1NYWV1b3yhulra2d3b3yvsHTS1TRWiDSC5VO8SaciZowzDDaTtRFMchp61wdDv1W09UaSbFgxknNIjxQLCIEWys1KzqM4RPe+WK53ozoGXi56QCOeq98le3L0kaU2EIx1p3fC8xQYaVYYTTSambappgMsID2rFU4JjqIJtdO0EnVumjSCpbwqCZ+nsiw7HW4zi0nTE2Q73oTcX/vE5qopsgYyJJDRVkvihKOTISTV9HfaYoMXxsCSaK2VsRGWKFibEBlWwI/uLLy6R57vpX7uX9RaXm5nEU4QiOoQo+XEMN7qAODSDwCM/wCm+OdF6cd+dj3lpw8plD+APn8wcZR44d</latexit>(s,a)Auxiliary NetworkSample-speciﬁc Inference (NCD)Event-speciﬁc Inference (Ours)Sample-Speciﬁc LCGsUnder review as a conference paper at ICLR 2024

(a)

(b)

(c)

(d)

(e)

(f)

(g)

Figure 9: Analysis of LCGs learned by our method with a codebook size of 4 in Chemical (full-fork)
environment. (a-c) Codebook histogram on (a) ID states, (b) ID states on local structure fork, and
(c) OOD states on local structure. (d-g) Learned LCGs. The descriptions of the histograms are also
applied to Figs. 10 to 12, 14 and 15.

(a)

(b)

(c)

(d)

(e)

(f)

(g)

Figure 10: Another sample run of our method with a codebook size of 4 in Chemical (full-fork).

the two codes, resulting in both codes corresponding to the global causal graph and failing to capture
the LCG, as shown in Fig. 15.

C.3.2 EVALUATION OF LOCAL CAUSAL DISCOVERY

The performance of our method and NCD in local causal discovery is evaluated using the Structural
Hamming Distance (SHD) in Magnetic. Structural Hamming Distance (SHD) is a metric used to
quantify the dissimilarity between two graphs based on the number of edge additions or deletions
needed to make the graphs identical (Acid & de Campos, 2003; Ramsey et al., 2006). As shown in
Fig. 16, our method consistently outperforms NCD across various codebook sizes except for K = 1,
where our method learns only a single causal graph over the entire data domain. In Fig. 16, SHD
scores are averaged over the data samples in the evaluation batch. For the samples in magnetic context
(i.e., both objects are red), we compare the inferred LCG with the global causal graph to measure
SHD. For the samples in non-magnetic context (i.e., one of the objects is black), we compare with the
one without redundant edges indicated with red boxes, as shown in Fig. 7(a). For example, as shown
in Fig. 5(a), our method with K = 1 infers a (global) causal graph correctly and shows the SHD
score of 6 in non-magnetic samples (Fig. 16, center) since inferred (global) causal graph includes
redundant edges in non-magnetic events (i.e., red boxes in Fig. 7(a)).

23

Under review as a conference paper at ICLR 2024

(a)

(b)

(c)

(d)

(e)

(f)

(g)

Figure 11: Analysis of LCGs learned by our method with a codebook size of 4 in Magnetic.

(a)

(b)

(c)

(d)

(e)

(f)

(g)

Figure 12: Another sample run of our method with a codebook size of 4 in Magnetic.

Figure 13: More fine-grained LCGs learned by our method with a codebook size of 16 in Magnetic.

(a)

(b)

C.3.3 ADDITIONAL DISCUSSIONS

Training with vector quantization. It is known that training discrete latent codebook with vector
quantization often suffers from the codebook collapsing, a well known issue in VQ-VAE literature.
As discussed in Appendix C.3.1, we observe similar behavior when training our method when K = 2.
Techniques have been recently proposed to prevent collapsing, such as codebook reset (Williams
et al., 2020) and stochastic quantization (Takida et al., 2022). We consider that incorporating such
techniques and tricks to further stabilize the training would be a future direction. We note that prior
works on learning discrete latent codebook mostly focused on the reconstruction of the observation

24

Under review as a conference paper at ICLR 2024

(a)

(b)

(c)

(d)

(e)

Figure 14: Analysis of LCGs learned by our method with a codebook size of 2 in Chemical (full-fork).

(a)

(b)

(c)

(d)

(e)

Figure 15: Failure case of our method with a codebook size of 2 in Chemical (full-fork).

Figure 16: Evaluation of local causal discovery of NCD and our method in Magnetic environment.

Figure 17: Learning curves on downstream tasks as measured on the average episode reward. Lines
and shaded areas represent the mean and standard deviation, respectively.

(Van Den Oord et al., 2017; Ozair et al., 2021) where the size of the codebook is much larger (e.g.,
128, 512, or 1024) and the utilization of vector quantization is quite different from ours.

Relationship with Hwang et al. (2023). Our work draws inspiration from Hwang et al. (2023),
which first discussed event-level decomposition. However, NCD, their proposed method, does not
explicitly discover such partitions but only infers LCG for each sample, as depicted in Fig. 8. In
, decomposition and corresponding LCGs. By
contrast, our method explicitly discovers
z
}
E
explicitly clustering samples into events and learning LCGs over each event, our method is more
robust on OOD states than sample-specific inference. When K = 1, our method is equivalent to
the score-based (global) causal discovery methods (Wang et al., 2021; Brouillard et al., 2020). As
K

, our method recovers NCD, a sample-specific inference method.

{G

z,

→ ∞

25

OursNCDSHD0246Number of codebook12481632ID (all)SHD0246Number of codebook12481632ID (non-magnetic)SHD0246Number of codebook12481632OOD (non-magnetic)MLPModularGNNNPSGRADEROracleNCDOurs00.51.01.5episode reward5101520Environment step (x105)Chemical (full-fork)00.51.01.5episode reward51015Environment step (x105)Chemical (full-chain)0.51.01.52.0episode reward0246810Environment step (x105)MagneticUnder review as a conference paper at ICLR 2024

Models

MLP

Modular

GNN

NPS

CDL

Grader

Oracle

NCD

Ours

Table 6: Parameters of each model.
Chemical

Magnetic

Parameters

Hidden dim
Hidden layers

Hidden dim
Hidden layers

Node attribute dim
Node network hidden dim
Node network hidden layers
Edge attribute dim
Edge network hidden dim
Edge network hidden layers

Number of rules
Cond selector dim
Rule embedding dim
Rule selector dim
Feature encoder hidden dim
Feature encoder hidden layers
Rule network hidden dim
Rule network hidden layers

Hidden dim
Hidden layers
CMI threshold
CMI optimization frequency
CMI evaluation frequency
CMI evaluation step size
CMI evaluation batch size
EMA discount

Feature embedding dim
GRU hidden dim
Causal discovery frequency

Hidden dim
Hidden layers

Hidden dim
Hidden layers
Auxiliary network hidden dim
Auxiliary network hidden layers

full-fork

full-chain

1024
3

1024
3

128
4

256
512
3
256
512
3

20
128
128
128
128
2
128
3

128
4
0.001
10
10
1
256
0.9

128
128
10

128
4

128
4
128
2

128
4

256
512
3
256
512
3

20
128
128
128
128
2
128
3

128
4
0.001
10
10
1
256
0.9

128
128
10

128
4

128
4
128
2

512
4

128
4

256
512
3
256
512
3

15
128
128
128
128
2
128
3

128
4
0.001
10
10
1
256
0.99

N/A
N/A
N/A

128
5

128
5
128
2

Hidden dim
Hidden layers
VQ encoder
VQ decoder
Codebook size
Code dimension

128
4
[128, 64]
[32]
16
16

128
4
[128, 64]
[32]
16
16

128
5
[128, 64]
[32]
16
16

C.3.4 LEARNING CURVES ON ALL DOWNSTREAM TASKS

Fig. 18 shows the learning curves on training in all environments. Figs. 17, 19 and 20 shows the
learning curves on all downstream tasks.6

C.3.5 LIMITATIONS AND FUTURE WORK.

Insufficient or biased data may lead to inaccurate learning of causal relationships. While we as-
sumed causal sufficiency, external factors or unobserved variables may also influence the causal
relationships. Future research directions include combining our method with explicit conditional
independence testing, and extending our framework to high-dimensional observation such as image,
where representation learning is crucial (Sch¨olkopf et al., 2021).

6As CDL is a two-stage method that requires searching the best threshold after the first stage training, we

only report their final performance.

26

Under review as a conference paper at ICLR 2024

Figure 18: Learning curves during training as measured by the episode reward.

Figure 19: Learning curves on downstream tasks in Chemical (full-fork) as measured on the episode
reward (top) and success rate (bottom).

Figure 20: Learning curves on downstream tasks in Chemical (full-chain) as measured on the episode
reward (top) and success rate (bottom).

27

MLPModularGNNNPSGRADEROracleNCDOurs00.51.01.5episode reward510152025Environment step (x105)Chemical (full-fork)00.51.01.5episode reward5101520Environment step (x105)Chemical (full-chain)0.51.01.52.0episode reward0510Environment step (x105)MagneticMLPModularGNNNPSGRADEROracleNCDOurs00.51.01.5episode reward5101520Environment step (x105)Chemical (full-fork) (n=2)00.51.01.5success ratio00.20.40.60.8Environment step (x105)00.51.01.5episode reward5101520Environment step (x105)Chemical (full-fork) (n=4)00.51.01.5success ratio00.20.40.60.8Environment step (x105)00.51.01.5episode reward5101520Environment step (x105)Chemical (full-fork) (n=6)00.51.01.5success ratio00.20.40.60.8Environment step (x105)MLPModularGNNNPSGRADEROracleNCDOurs00.51.01.5episode reward51015Environment step (x105)Chemical (full-chain) (n=2)00.51.01.5success ratio00.20.40.6Environment step (x105)00.51.01.5episode reward51015Environment step (x105)Chemical (full-chain) (n=4)00.51.01.5success ratio00.20.40.6Environment step (x105)00.51.01.5episode reward51015Environment step (x105)Chemical (full-chain) (n=6)00.51.01.5success ratio00.20.40.60.81.0Environment step (x105)