Under review as a conference paper at ICLR 2023

MODEL TRANSFERABILITY WITH RESPONSIVE DECI-
SION SUBJECTS

Anonymous authors
Paper under double-blind review

ABSTRACT

D

This paper studies model transferability when human decision subjects respond to
a deployed machine learning model. In our setting, an agent or a user corresponds
to a sample (X, Y ) drawn from a distribution
and will face a model h and
its classiﬁcation result h(X). Agents can modify X to adapt to h, which will
incur a distribution shift on (X, Y ). Therefore, when training h, the learner will
need to consider the subsequently “induced” distribution when the output model
is deployed. Our formulation is motivated by applications where the deployed
machine learning models interact with human agents, and will ultimately face
responsive and interactive data distributions. We formalize the discussions of the
transferability of a model by studying how the model trained on the available source
distribution (data) would translate to the performance on the induced domain. We
provide both upper bounds for the performance gap due to the induced domain shift,
as well as lower bounds for the trade-offs that a classiﬁer has to suffer on either the
source training distribution or the induced target distribution. We provide further
instantiated analysis for two popular domain adaptation settings with covariate
shift and target shift.

1

INTRODUCTION

Decision makers are increasingly required to be transparent on their decision making to offer the
“right to explanation” (Goodman & Flaxman, 2017; Selbst & Powles, 2018; Ustun et al., 2019) 1.
Being transparent also invites potential adaptations from the population, leading to potential shifts.
We are motivated by settings where the deployed machine learning models interact with human
agents, which will ultimately face data distributions that reﬂect how human agents respond to the
models. For instance, when a model is used to decide loan applications, candidates may adapt their
features based on the model speciﬁcation in order to maximize their chances of approval; thus the
loan decision classiﬁer observes a data distribution caused by its own deployment (e.g., see Figure 1
for a demonstration). Similar observations can be articulated for application in insurance sector (i.e.
developing policy s.t. customers’ behaviors might adapt to lower premium (Haghtalab et al., 2020)),
education sector (i.e. developing courses when students are less incentivized to cheat (Kleinberg &
Raghavan, 2020)) and so on.

FEATURE

Income

Education Level

Debt

Savings

WEIGHT ORIGINAL VALUE

ADAPTED VALUE

2

3

-10

5

$ 6,000

College

$40,000

$20,000

−→

−→

−→

−→

$ 6,000

College

$20,000

$0

Figure 1: An example of an agent who originally has both savings and debt, observes that the classiﬁer
penalizes debt (weight -10) more than it rewards savings (weight +5), and concludes that their most
efﬁcient adaptation is to use their savings to pay down their debt.

This paper investigates model transferability when the underlying distribution shift is induced by the
deployed model. What we would like to have is some guarantee on the transferability of a classiﬁer —

1See Appendix A.1 for more detailed discussions.

1

Under review as a conference paper at ICLR 2023

that is, how training on the available source distribution
domain
induced risk, deﬁned as the error a model incurs on the distribution induced by itself:

S translates to performance on the induced
(h), which depends on the model h being deployed. A key concept in our setting is the

D

D

Induced Risk : ErrD(h)(h) := PD(h)(h(X)

= Y )

(1)

Most relevant to the above formulation is the strategic classiﬁcation literature (Hardt et al., 2016a;
Chen et al., 2020b). In this literature, agents are modeled as rational utility maximizers and game
theoretical solutions were proposed to characterize the induced risk. However, our results are
motivated by the following challenges in more general scenarios:

•

•

•

Modeling assumptions being restrictive In many practical situations, it is often hard to faithfully
characterize agents’ utilities. Furthermore, agents might not be fully rational when they response.
All the uncertainties can lead to a far more complicated distribution change in (X, Y ), as compared
to often-made assumptions that agents only change X but not Y (Chen et al., 2020b).
Lack of access to response data Another relevant literature to our work is performative prediction
(h) or
(Perdomo et al., 2020). In performative prediction, one would often require knowing
(h) through repeated experiments. We posit that machine learning
having samples observed from
practitioners may only have access to data from the source distribution during training, and although
they anticipate changes in the population due to human agents’ responses, they cannot observe this
new distribution until the model is actually deployed.
Retraining being costly Even when samples from the induced data distribution are available,
retraining the model from scratch may be impractical due to computational constraints.

D

D

The above observations motivate us to understand the transferability of a model trained on the source
data to the domain induced by the deployment of itself. We study several fundamental questions:

•

•

•

•

⇒

⇒

= Y ), the error on the source?

Minimum induced risk How much higher is ErrD(h)(h), the error on the

Induced risk For a given model h, how different is ErrD(h)(h), the error on the

Source risk
distribution induced by h, from ErrDS (h) := PDS (h(X)
Induced risk
induced distribution, than minh(cid:48) ErrD(h(cid:48))(h(cid:48)), the minimum achievable induced error?
Induced risk of source optimal
case of the above, how does ErrD(h∗
source distribution h∗
Lower bound for learning tradeoffs What is the minimum error a model must incur on either
the source distribution ErrDS (h) or its induced distribution ErrD(h)(h)?

Minimum induced risk Of particular interest, and as a special
S )(h∗
S), the induced error of the optimal model trained on the

S := arg minh ErrDS (h), compare to h∗

T := arg minh ErrD(h)(h)?

⇒

For the ﬁrst three questions, we prove upper bounds on the additional error incurred when a model
trained on a source distribution is transferred over to its induced domain. We also provide lower
bounds for the trade-offs a classiﬁer has to suffer on either the source training distribution or the
induced target distribution. We then show how to specialize our results to two popular domain
adaptation settings: covariate shift (Shimodaira, 2000; Zadrozny, 2004; Sugiyama et al., 2007; 2008;
Zhang et al., 2013b) and target shift (Lipton et al., 2018; Guo et al., 2020; Zhang et al., 2013b). All
omitted proofs can be found in the Appendix.

1.1 RELATED WORKS
Most relevant to us are three topics: strategic classiﬁcation (Hardt et al., 2016a; Chen et al., 2020b;
Dekel et al., 2010; Dong et al., 2018; Chen et al., 2020a; Miller et al., 2020; Kleinberg & Raghavan,
2020), a recently proposed notion of performative prediction (Perdomo et al., 2020; Mendler-D¨unner
et al., 2020), and domain adaptation (Jiang, 2008; Ben-David et al., 2010; Sugiyama et al., 2008;
Zhang et al., 2019; Kang et al., 2019; Zhang et al., 2020).

Hardt et al. (2016a) pioneered the formalization of strategic behavior in classiﬁcation based on
a sequential two-player game between agents and classiﬁers. Subsequently, Chen et al. (2020b)
addressed the question of repeatedly learning linear classiﬁers against agents who are strategically
trying to game the deployed classiﬁers. Most of the existing literature focuses on ﬁnding the optimal
classiﬁer by assuming fully rational agents (and by characterizing the equilibrium response). In
contrast, we do not make these assumptions and primarily study the transferability when only having
knowledge of source data.

2

(cid:54)
(cid:54)
Under review as a conference paper at ICLR 2023

Our result was inspired by the transferability results in domain adaptations (Ben-David et al., 2010;
Crammer et al., 2008; David et al., 2010). Later works examined speciﬁc domain adaptation models,
such as covariate shift (Shimodaira, 2000; Zadrozny, 2004; Gretton et al., 2009; Sugiyama et al.,
2008; Zhang et al., 2013b;a) and target/label shift (Lipton et al., 2018; Azizzadenesheli et al., 2019).
A commonly established solution is to perform reweighted training on the source data, and robust and
efﬁcient solutions have been developed to estimate the weights accurately (Sugiyama et al., 2008;
Zhang et al., 2013b;a; Lipton et al., 2018; Guo et al., 2020).

Our work, at the ﬁrst sight, looks similar to several other area of studies. For instance, the notion of
observing an “induced distribution” resembles similarity to the adversarial machine learning literature
(Lowd & Meek, 2005; Huang et al., 2011; Vorobeychik & Kantarcioglu, 2018). One of the major
differences between us and adversarial machine learning is the true label Y stays the same for the
attacked feature while in our paper, both X and Y might change in the adapted distribution
(h). In
Appendix A.2, we provide detailed comparisons with some areas in domain adaptations, including
domain generalization, adversarial attack and test-time adaptation. In particular, similar to domain
generalization, one of the biggest challenge for our setting is the lack of access to data from the target
distribution during training.

D

2 FORMULATION

xi, yi
{

Suppose we are learning a parametric model h
for a binary classiﬁcation problem. Its training
∈ H
N
i=1 is drawn from a source distribution
data set S :=
1, +1
.
}
}
However, h will then be deployed in a setting where the samples come from a test or target distribution
S. Therefore instead of minimizing the prediction error
T that can differ substantially from
= Y ), the goal is to ﬁnd h∗ that minimizes
= Y ). This is often referred to as the domain adaptation problem, where
T is assumed to be independent of the model h being deployed.

D
on the source distribution ErrDS (h) := PDS (h(X)
ErrDT (h) := PDT (h(X)
typically, the transition from

S, where xi

Rd and yi

∈ {−

S to

D

D

∈

D

D

We consider a setting in which the distribution shift depends on h, or is thought of as being induced
by h. We will use

(h) to denote the induced domain by h:

D

S

encounters model h

(h)

D
Strictly speaking, the induced distribution is a function of both
by
of
we will further instantiate

S(h). To ease the notation, we will stick with
S. For now, we do not restrict the dependency of

D
(h) under speciﬁc domain adaptation settings.

D
(h) of

S and h and should be better denoted
(h), but we shall keep in mind of its dependency
and h, but later in Section 4 and 5

→ D

D
D

→

D

D

The challenge in the above setting is that when training h, the learner needs to carry the thoughts that
(h) should be the distribution it will be evaluated on and that the training cares about. Formally, we

D
deﬁne the induced risk of a classiﬁer h as the 0-1 error on the distribution h induces:

D

Induced risk :

ErrD(h)(h) := PD(h)(h(X)

= Y )

(2)

Denote by h∗
when the loss may not be the 0-1 loss, we deﬁne the induced (cid:96)-risk as

T := arg minh∈H ErrD(h)(h) the classiﬁer with minimum induced risk. More generally,

Induced (cid:96)-risk :

Err(cid:96),D(h)(h) := Ez∼D(h)[(cid:96)(h; z)]

The induced risks will be the primary quantities that we are interested in minimizing. The following
additional notation will also be helpful:

:

D

Distributions of Y on a distribution
D
DY |S := PDS (Y = y).
PD(h)(Y = y),
Distribution of h on a distribution
:
D
PD(h)(h(X) = y),
Marginal distribution of X for a distribution
DX|S := PDS (X = x)3.
PD(h)(X = x),
Total variation distance (Ali & Silvey, 1966): dTV(

Dh|S := PDS (h(X) = y).
D

D

D

:

•

•

•

•

Y := PD(Y = y)2, and in particular

h := PD(h(X) = y), and in particular

X := PD(X = x), and in particular

Y (h) :=

D

h(h) :=

D

X (h) :=

D

(cid:48)) := supO|
2The “:=” deﬁnes the RHS as the probability measure function for the LHS.
3For continuous X, the probability measure shall be read as the density function.

D

D

O

,

PD(

)

−

PD(cid:48)(

O

.
)
|

3

(cid:54)
(cid:54)
(cid:54)
Under review as a conference paper at ICLR 2023

2.1 EXAMPLES OF DISTRIBUTION SHIFTS INDUCED BY MODEL DEPLOYMENT

We provide two example models to demonstrate the use cases for the distribution shift models
described in our paper. We provide more details in Section 4.3 and Section 5.3.

Strategic Classiﬁcation An example of distribution shift is when decision subjects perform strate-
gic response to a decision rule. It is well-known that when human agents are subject to a decision
rule, they will adapt their features so as to get a favorable prediction outcome. In the literature of
strategic classiﬁcation, we say the human agents perform strategic adaptation (Hardt et al., 2016a).

It is natural to assume that the feature distribution before and after the human agents’ best response
satisﬁes covariate shift: namely the feature distribution P(X) will change, but P(Y
X), the mapping
between Y and X, remain unchanged. Notice that this is different from the assumption made in the
classic strategic classiﬁcation setting Hardt et al. (2016a), which is to assume that the changes in the
feature X does not change the underlying true qualiﬁcation Y . In our paper, we assume that changes
in feature X could potential lead to changes in the true qualiﬁcation Y , and that the mapping between
Y and X remains the same before and after the adaptation. This is a commonly assumption made in
a recent line of work on incentivizing improvement behaviors from human agents(see, e.g. Chen et al.
(2020a); Shavit et al. (2020)). We use Figure 2 (Left) as a demonstration of how distribution might
shift for strategic response setting. In Section 4.3, we will use the strategic classiﬁcation setup to
verify our obtained results.

|

Figure 2: Example causal graph annotated to demonstrate covariate shift (Left) / target shift (Right)
as a result of the deployment of h. Grey nodes indicate observable variables and transparent nodes
are not observed at the training stage. Red arrow emphasises h induces changes of certain variables.

Replicator Dynamics Replicator dynamics is a commonly used model to study the evolution of an
adopted “strategy” in evolutionary game theory (Tuyls et al., 2006; Friedman & Sinervo, 2016; Taylor
& Jonker, 1978; Raab & Liu, 2021). The core notion of it is the growth or decline of the population
of each strategy depends on its “ﬁtness”. Consider the label Y =
as the strategy, and the
following behavioral response model to capture the induced target shift:

1, +1

{−

}

PD(h)(Y = +1)
PDS (Y = +1)

=

Fitness(Y = +1)
EDS [Fitness(Y )]

In short, the change of the Y = +1 population depends on how predicting Y = +1 “ﬁts” a certain
utility function. For instance, the “ﬁtness” can take the form of the prediction accuracy of h for
class +1: Fitness(Y = +1) := PDS (h(X) = +1
Y = +1) . Intuitively speaking, a higher “ﬁtness”
|
1 or Y = +1). Therefore,
describes more success of agents who adopted a certain strategy (Y =
agents will imitate or replicate these successful peer agents by adopting the same strategy, resulting in
an increase of the population (PD(h)(Y )). With assuming P(X
Y ) stays unchanged, this instantiates
one example of a speciﬁc induced target shift. We will specify the condition for target shift in
Section 5. We use Figure 2 (Right) as a demonstrating of how distribution might shift for the
replicator dynamic setting. In Section 5.3, we will use a detailed replicator dynamics model to further
instantiate our results.

−

|

3 TRANSFERABILITY OF LEARNING TO INDUCED DOMAINS

In this section, we ﬁrst provide upper bounds for the transfer error of a classiﬁer h (that is, the
difference between ErrD(h)(h) and ErrDS (h)), as well as between ErrD(h)(h) and ErrD(h∗
T ). We
, that is, the minimum error a model h
then provide lower bounds for max
}
(h).
S or the induced distribution
must incur on either the source distribution

ErrDS (h), ErrD(h)(h)
{

T )(h∗

D

D

4

X1X1X2X2X3X3YYh(X)h(X)X′ 1X′ 1X′ 2X′ 2X′ 3X′ 3Y′ Y′ YYX1X1 X3X3X2X2h(X)h(X)Y′ Y′ X′ 1X′ 1X′ 3X′ 3X′ 2X′ 2Under review as a conference paper at ICLR 2023

3.1 UPPER BOUND
We ﬁrst investigate upper bounds for the transfer errors. We begin by showing generic bounds, and
further instantiate the bound for speciﬁc domain adaptation settings in Section 4 and 5 . We begin
with answering a central question in domain adaptation:

How does a model h trained on its training data set fare on the induced distribution

To that end, deﬁne the minimum and h-dependent combined error of two distributions

D
ErrD(cid:48)(h(cid:48)) + ErrD(h(cid:48)), ΛD→D(cid:48)(h) := ErrD(cid:48)(h) + ErrD(h)

λD→D(cid:48) := min
h(cid:48)∈H

(h)?

D
and

(cid:48) as:

D

,

D

D

H
H

(cid:48)) = 2 suph,h(cid:48)∈H |

-divergence as dH×H(
.
and
|
The
-divergence is celebrated measure proposed in the domain adaptation literature (Ben-David
et al., 2010) which will be useful for bounding the difference in errors of two classiﬁers. Repeating
classical arguments from Ben-David et al. (2010), we can easily prove the following:
Theorem 3.1 (Source risk
⇒
upper bounded by: ErrD(h)(h)

Induced risk). The difference between ErrD(h)(h) and ErrDS (h) is
2 dH×H(

ErrDS (h) + λDS→D(h) + 1

= h(cid:48)(X))

= h(cid:48)(X))

PD(cid:48)(h(X)

PD(h(X)

(h)).

S,

−

≤

D

D

T : ErrD(h)(h)

The transferability of a model h between ErrD(h)(h) and ErrDS (h) looks precisely the same as in
the classical domain adaptation setting (Ben-David et al., 2010). Nonetheless, an arguably more
interesting quantity in our setting to understand is the difference between the induced error of a given
model h and the error induced by the optimal model h∗
T ). We get the
following bound, which differs from the one in Theorem 3.1:
Theorem 3.2 (Induced risk

Minimum induced risk). The difference between ErrD(h)(h) and

⇒
T ) is upper bounded by: ErrD(h)(h)
(h∗

ErrD(h∗
1
2 ·
The above theorem informs us that the induced transfer error is bounded by the “average” achievable
error on both distributions
divergence between the two
distributions. Reﬂecting on the difference between the bounds of Theorem 3.1 and Theorem 3.2, we
see that the primary change is replacing the minimum achievable error λ with the average of λ and Λ.

T ), as well as the

)+ΛD(h)→D(h∗
T

λD(h)→D(h∗
T

T )(h∗
T )

ErrD(h∗

ErrD(h∗

(h) and

T )(h∗

T )(h∗

dH×H(

H × H

(h)).

(h∗

T ),

)(h)

−

−

≤

D

D

D

D

+

2

3.2 LOWER BOUND
Now we provide a lower bound on the induced transfer error. We particularly want to show that at
least one of the two errors ErrDS (h), and ErrD(h)(h), must be lower-bounded by a certain quantity.
Theorem 3.3 (Lower bound for learning tradeoffs ). Any model h must
incur the fol-
lowing error on either the source or induced distribution: max
dTV(DY |S ,DY (h))−dTV(Dh|S ,Dh(h))
2

ErrDS (h), ErrD(h)(h)
{

} ≥

.

The proof leverages the triangle inequality of dTV. This bound is dependent on h; however, by the
data processing inequality of dTV (and f -divergence functions in general) (Liese & Vajda, 2006), we
have dTV(

X (h)). Applying this to Theorem 3.3 yields:

h(h))

Dh|S,

DX|S,
Corollary 3.4. For any model h, max
{

dTV(

≤

D

D
ErrDS (h), ErrD(h)(h)

} ≥

dTV(DY |S ,DY (h))−dTV(DX|S ,DX (h))
2

.

3.3 HOW TO USE OUR BOUNDS

The upper and lower bounds we derived in the previous sections (Theorem 3.2 and Theorem 3.3)
(h) induced
depend on the following two quantities either explicitly or implicitly: 1) the distribution
by the deployment of the model h in question, and 2) the optimal target classiﬁer h∗
T as well as the
T ) it induces. The bounds may therefore seem to be of only theoretical interest, since
distribution
in reality we generally cannot compute
T . Thus
in general it is unclear how to compute the value of these bounds. Nevertheless, our bounds can still
be useful and informative in the following ways:

(h) without actual deployment, let alone compute h∗

(h∗

D

D

D

General modeling framework with ﬂexible hypothetical shifting models The bounds can be
evaluated if the decision maker has a particular shift model in mind, which speciﬁes how the
population would adapt to a model. A common special case is when the decision maker posits an

5

(cid:54)
(cid:54)
Under review as a conference paper at ICLR 2023

individual-level agent response model (e.g. the strategic agent (Hardt et al., 2016a) - we demonstrate
how to evaluate in Section 4.3). In these cases, the H-divergence can be consistently estimated from
ﬁnite samples of the population (Wang et al., 2005), allowing the decision maker to estimate the
performance gap of a given h without deploying it. The general bounds provided can thus be viewed
as a framework by which specialized, computationally tractable bounds can be derived.

Estimate the optimal target classiﬁer h∗
decision maker has access to a set of imperfect models ˜h1, ˜h2
range of possible shifted distribution
distribution hT
∈ H
this predicted set 4:

T from a set of imperfect models Secondly, when the
˜ht
H T that will predict a
(˜h1),
T and a range of possibly optimal target
T can be further instantiated by calculating the worst case in

D
T , the bounds on h∗

(˜ht)

· · · D

∈ D

· · ·

∈

ErrD(h)(h)

ErrD(h∗

−
ErrDS (h), ErrD(h∗

T )(h∗
T )(h∗
T )

T ) (cid:46)

max
D(cid:48)∈DT ,h(cid:48)∈HT

UpperBound(

(cid:38)

}

min
D(cid:48)∈DT ,h(cid:48)∈HT

LowerBound(

max
{

(cid:48), h(cid:48)),

(cid:48), h(cid:48)).

D

D

In addition, the challenge we are facing in this paper also shed lights on the danger of directly applying
existing standard domain adaptation techniques when the shifting is caused by the deployment of the
classiﬁer itself, since the bound will depend on the resulting distribution as well. We add discussions
on the tightness of our theoretical bounds in Appendix G.

4 COVARIATE SHIFT

In this section, we focus on a particular domain adaptation setting known as covariate shift, in which
the distribution of features changes, but the distribution of labels conditioned on features does not:

PD(h)(Y = y

|

X = x) = PDS (Y = y

|

X = x), PD(h)(X = x)

= PDS (X = x)

(3)

Thus with covariate shift, we have

PD(h)(X = x, Y = y) =PD(h)(Y = y

X = x)
|

·

PD(h)(X = x) = PDS (Y = y

X = x)
|

·

PD(h)(X = x)

Let ωx(h) :=
tion induced by h at instance x. Then for any loss function (cid:96) we have

PD(h)(X=x)
PDS (X=x) be the importance weight at x, which characterizes the amount of adapta-

Proposition 4.1 (Expected Loss on

(h)). ED(h)[(cid:96)(h; X, Y )] = EDS [ωx(h)

D

(cid:96)(h; x, y)].

·

The above derivation was not new and offered the basis for performing importance reweighting when
learning under coviarate shift (Sugiyama et al., 2008). The particular form informs us that ωx(h)
S and h, and is critical for
(h) and encodes its dependency of both
controls the generation of
deriving our results below.

D

D

4.1 UPPER BOUND

We now derive an upper bound for transferability under covariate shift. We will focus particularly on
S := arg minh∈H ErrS(h).
the optimal model trained on the source data
D
Recall that the classiﬁer with minimum induced risk is denoted as h∗
T := arg minh∈H ErrD(h)(h).
We can upper bound the difference between h∗
S and h∗
Theorem 4.2 (Suboptimality of h∗

S, which we denote as h∗

S). Let X be distributed according to

T as follows:

S. We have:

D

ErrD(h∗

S )(h∗
S)

−

ErrD(h∗

T )(h∗
T )

≤

(cid:113)

ErrDS (h∗
T )

·

(cid:18)(cid:113)

Var(ωX (h∗

S)) +

(cid:113)

(cid:19)

Var(ωX (h∗

T ))

.

This result can be interpreted as follows: h∗
T incurs an irreducible amount of error on the source
data set, represented by (cid:112)ErrDS (h∗
T is at its
maximum when the two classiﬁers induce adaptations in “opposite” directions; this is represented by
the sum of the standard deviations of their importance weights, (cid:112)Var(ωX (h∗
T )).

S and h∗
S)) + (cid:112)Var(ωX (h∗

T ). Moreover, the difference in error between h∗

4UpperBound and LowerBound are the RHS expressions in Theorem 3.3 and Theorem 3.2.

6

(cid:54)
Under review as a conference paper at ICLR 2023

4.2 LOWER BOUND

Recall from Theorem 3.3, for the general setting, it is unclear whether the lower bound is strictly
In this section, we provide further understanding for when the lower bound
positive or not.
dTV(DY |S ,DY (h))−dTV(Dh|S ,Dh(h))
is indeed positive under covariate shift. Under several assump-
2

tions, our previously provided lower bound in Theorem 3.3 is strictly positive with covariate shift.
Assumption 4.3.

EX∈X−(h),Y =+1[1

ωX (h)]

.

ωX (h)]
|

−

EX∈X+(h),Y =+1[1
|
x : ωx(h)
{

≥

}

1

−

| ≥ |

and X−(h) =

x : ωx(h) < 1
}
{

.

where X+(h) =

EX∈X+(h),h(X)=+1[1
|

This assumption states that increased ωx(h) value points are more likely to have positive labels.
ωX (h)]
Assumption 4.4.
This assumption states that increased ωx(h) value points are more likely to be classiﬁed as positive.
Assumption 4.5. Cov(cid:0)PDS (Y = +1
X = x)
|
This assumption is stating that for a classiﬁer h, within all h(X) = +1 or h(X) =
PD(Y = +1
|
Theorem 4.6. Assuming 4.3 - 4.5, the following lower bound is strictly positive for covariate shift:

X = x), ωx(h)(cid:1) > 0.
PDS (h(x) = +1
|

X = x) associates with a higher ωx(h).

EX∈X−(h),h(X)=+1[1

ωX (h)]
.
|

1, a higher

| ≥ |

−

−

−

−

max

ErrDS (h), ErrD(h)(h)
{

} ≥

dTV(

Y |S,

D

Y (h))

D

−
2

dTV(

h|S,

D

D

h(h))

> 0.

4.3 EXAMPLE USING STRATEGIC CLASSIFICATION

·

∈

τh]

1[x

h(x)

∈ {−

Rd and a binary true qualiﬁcation y(x)

As introduced in Section 2.1, we consider a setting caused by strategic response in which agents are
classiﬁed by and adapt to a binary threshold classiﬁer. In particular, each agent is associated with a
d dimensional continuous feature x
1, +1
, where
}
y(x) is a function of the feature vector x. Consistent with the literature in strategic classiﬁcation
(Hardt et al., 2016a), a simple case where after seeing the threshold binary decision rule h(x) =
1, the agents will best response to it by maximizing the following utility function:
2
−
≥
u(x, x(cid:48)) = h(x(cid:48))
c(x, x(cid:48)), where c(x, x(cid:48)) is the cost function for decision subjects to modify
their feature from x to x(cid:48). We assume all agents are rational utility maximizers: they will only
attempt to change their features when the beneﬁt of manipulation is greater than the cost (i.e. when
c(x, x(cid:48))
2) and agent will not change their feature if they are already accepted (i.e. h(x) = +1).
For a given threshold τh and manipulation budget B, the theoretical best response of an agent with
original feature x is: ∆(x) = arg maxx(cid:48) u(x, x(cid:48)) s.t. c(x, x(cid:48))
B. To make the problem tractable
and meaningful, we further specify the following setups:
Setup 1. (Initial Feature) Agents’ initial features are uniformly distributed between [0, 1]
Setup 2. (Agent’s Cost Function) The cost of changing from x to x(cid:48) is proportional to the distance
between them: c(x, x(cid:48)) =

R1.

x(cid:48)

≤

−

−

≤

∈

x
(cid:107)

−

.
(cid:107)

B, τh) will attempt to change
Setup 2 implies that only agents whose features are in between [τh
their feature. We also assume that feature updates are probabilistic, such that agents with features
closer to the decision boundary τh have a greater chance of updating their feature and each updated
feature x(cid:48) is sampled from a uniform distribution depending on τh, B, and x (see Setup 3 & 4):
Setup 3. (Agent’s Success Manipulation Probability) For agents who attempt to update their features,
the probability of a successful feature update is P(X (cid:48)
Setup 4 (Adapted Feature’s Distribution). An agent’s updated feature x(cid:48), given original x, manipula-
tion budget B, and classiﬁcation boundary τh, is sampled as X (cid:48)

Unif(τh, τh +

= X) = 1

|x−τh|
B .

−

−

B

x

∼

|

−

).
|

Setup 4 aims to capture the fact that even though agent targets to change their feature to the decision
boundary τh (i.e. the least cost action to get a favorable prediction outcome), they might end up
reaching to a feature that is beyond the decision boundary. With the above setups, we can specify the
bound in Theorem 4.2 for the strategic response setting as follows:

Proposition 4.7 (Strategic Response Setting). ErrD(h∗

S )(h∗
S)

ErrD(h∗

T )(h∗
T )

−

≤

(cid:113) 2B

3 ErrDS (h∗

T ).

We can see that the upper bound for strategic response depends on the manipulation budget B, and
the error the ideal classiﬁer made on the source distribution ErrDS (h∗
T ). This aligns with our intuition

7

(cid:54)
Under review as a conference paper at ICLR 2023

that the smaller manipulation budget is, the less agents will change their features, thus leading to a
T ). This bound also allows us
tighter upper bound on the difference between Errh∗
(h) and h, since we
to bound this quantity even without the knowledge of the mapping between
T ) from the source distribution and an estimated optimal classiﬁer h∗
can directly compute ErrDS (h∗
T .

S) and Errh∗

T (h∗

S (h∗

D

5 TARGET SHIFT

We consider another popular domain adaptation setting known as target shift, in which the distribution
of labels changes, but the distribution of features conditioned on the label remains the same:

PD(h)(X = x

Y = y) = PDS (X = x
|

Y = y), PD(h)(Y = y)
|

= PDS (Y = y)

(4)

p(h). Again,
S and h. Then we have for any proper loss function (cid:96):

For binary classiﬁcation, let p(h) := PD(h)(Y = +1), and PD(h)(Y =
p(h) encodes the induced adaptation from
ED(h)[(cid:96)(h; X, Y )] =p(h)
=p(h)

D
ED(h)[(cid:96)(h; X, Y )
ED(h)[(cid:96)(h; X, Y )
Y = +1] + (1
Y =
|
|
·
EDS [(cid:96)(h; X, Y )
EDS [(cid:96)(h; X, Y )
Y = +1] + (1
Y =
|
|
We will adopt the following shorthands: Err+(h) := EDS [(cid:96)(h; X, Y )
Y = +1], Err−(h) :=
|
EDS [(cid:96)(h; X, Y )
1]. Note that Err+(h), Err−(h) are both deﬁned on the conditional source
|
distribution, which is invariant under the target shift assumption.

−
p(h))

1) = 1

p(h))

−
1]

Y =

−

−

−

−

−

1]

·

·

·

5.1 UPPER BOUND

We again upper bound the transferability of h∗
label distribution on
(PDS (X = x
Theorem 5.1. For target shift, the difference between ErrD(h∗

D
1)). Let p := PDS (Y = +1).

S (PDS (X = x

Y = +1)) and

Y =

−

D

|

|

S under target shift. Denote by

− the negative label distribution on

+ the positive
S

D

D

ErrD(h∗

S )(h∗
S)

ErrD(h∗

T )(h∗
T )

p(h∗
S)

p(h∗
T )

|

−

≤ |

−

+ (1 + p)

·

(dTV(

D

S )(h∗

S) and ErrD(h∗
+(h∗

+(h∗

T )(h∗
T )) + dTV(

T ) bounds as:
−(h∗

S),

S),

D

D

−(h∗

T )) .

D

The above upper bound consists of two components. The ﬁrst quantity captures the difference
(h∗
T ). The second quantity characterizes the
between the two induced distributions
D
S, h∗
difference between the two classiﬁers h∗

S) and
T on the source distribution.

(h∗

D

5.2 LOWER BOUND

Now we discuss lower bounds. Denote by TPRS(h) and FPRS(h) the true positive and false positive
rates of h on the source distribution
Theorem 5.2. For target shift, any model h must incur the following error on either

S. We prove the following:

D

(h):

S or

D

D

max

ErrDS (h), ErrD(h)(h)
{

} ≥

p
|

−

p(h)

| ·

(1

− |

TPRS(h)

2

FPRS(h)
|

)

.

−

D

Dh|S,

h(h)) under the assumption of target shift. Since

The proof extends the bound of Theorem 3.3 by further explicating each of dTV(
Y (h)),
DY |S,
D
dTV(
< 0 unless we
FPRS(h)
|
have a trivial classiﬁer that has either TPRS(h) = 1, FPRS(h) = 0 or TPRS(h) = 0, FPRS(h) = 1,
the lower bound is strictly positive. Taking a closer look, the lower bound is determined linearly
p(h). The difference is further determined by the
by how much the label distribution shifts: p
performance of h on the source distribution through 1
FPRS(h)
. For instance, when
|
TPRS(h) > FPRS(h), the quality becomes FNRS(h) + FPRS(h), that is the more error h makes,
the larger the lower bound will be.

TPRS(h)
|

TPRS(h)

− |

−

−

−

5.3 EXAMPLE USING REPLICATOR DYNAMICS

Let us instantiate the discussion using a speciﬁc ﬁtness function for the replicator dynamics model
(Section 2.1), which is the prediction accuracy of h for class y:
Fitness(Y = y) := PDS (h(X) = y

Y = y)

(5)

|

PrDS (h(X)=+1|Y =+1)
1−ErrDS (h)

. Plugging

Then we have E [Fitness(Y )] = 1
the result back to our Theorem 5.1 we have

−

ErrDS (h), and

p(h)

PDS (Y =+1) =

8

(cid:54)
Under review as a conference paper at ICLR 2023

Figure 3: Diff := ErrD(h∗
T )}, UB := upper bound speciﬁed
in Theorem 4.2, and LB := lower bound speciﬁed in Theorem 4.6. For each time step K = k, we compute and deploy the source optimal
classiﬁer h∗
S and update the credit score for each individual according to the received decision as the new reality for time step K = k + 1.
Details of the data generation is again deferred to Appendix C.

T ), Max := max{ErrDS (h∗

S ) − ErrD(h∗
T

T ), ErrD(h∗
T

)(h∗

)(h∗

)(h∗

S

Proposition 5.3. Under the replicator dynamics model in Eqn. (21),

ω(h∗
S)
|

−

ω(h∗
T )

| ≤

PDS (Y = +1)

·

ErrDS (h∗
S)
|
(1

ErrDS (h∗
T )
ErrDS (h∗
S))

−
−

bounds as:

ω(h∗
S)
−
|
TPRS(h∗
S)
| · |
−
ErrDS (h∗
(1
·

−

ω(h∗

T )
|
TPRS(h∗
T ))

T )

.

|

That is, the difference between ErrD(h∗
between the two classiﬁers’ performances on the source data
evaluate the possible error transferability using the source data only.

S) and ErrD(h∗

D

T )(h∗

S )(h∗

T ) is further dependent on the difference
S. This offers an opportunity to

6 EXPERIMENTS

We perform synthetic experiments using real-world data to demonstrate our bounds. In particular,
we use the FICO credit score data set (Board of Governors of the Federal Reserve System (US),
2007) which contains more than 300k records of TransUnion credit score of clients from different
demographic groups. For our experiment on the preprocessed FICO data set (Hardt et al., 2016b),
we convert the cumulative distribution function (CDF) of TransRisk score among different groups
into group-wise credit score densities, from which we generate a balanced sample to represent a
population where groups have equal representations. We demonstrate the application of our results in
a series of resource allocations. We consider the hypothesis class of threshold classiﬁers and treat the
classiﬁcation outcome as the decision received by individuals.
For each time step K = k, we compute h∗
S, the statistical optimal classiﬁer on the source distribution
(i.e., the current reality for step K = k), and update the credit score for each individual according
to the received decision as the new reality for time step K = k + 1. Details of the data generation
is again deferred to Appendix C. We report our results in Figure 3. We do observe positive gaps
ErrD(h∗
S. The gaps are well
bounded by the theoretical upper bound (UB). Our lower bounds (LB) do return meaningful positive
gaps, demonstrating the trade-offs that a classiﬁer has to suffer on either the source distribution or the
induced target distribution.

T ), indicating the suboptimality of training on

S )(h∗
S)

ErrD(h∗

T )(h∗

−

D

Challenges in Minimizing Induced Risk and Concluding Remarks We presented a sequence
of model transferability results for settings where agents will respond to a deployed model. The
response leads to an induced distribution that the learner would not know before deploying the model.
Our results cover for both a general response setting and for speciﬁc ones (covariate shift and target
shift). Looking forward to solving the induced risk minimization, the literature of domain adaptation
has provided us solutions to minimize the risk on the target distribution via a nicely developed set of
results (Sugiyama et al., 2008; 2007; Shimodaira, 2000). This allows us to extend the solutions to
minimize the induced risk too. Nonetheless we will highlight additional computational challenges.
Let’s use the covariate shift setting. The scenario for target shift is similar. For covariate shift, recall
that earlier we derived the following fact:

(Importance Reweighting) : ED(h)[(cid:96)(h; X, Y )] = ED[ωx(h)

(cid:96)(h; x, y)] .

·

(6)

This formula informs us that a promising solution that uses ωx(h) to perform reweighted ERM. There
are two primary challenges when carrying out optimization of the above objective. Of course, the
primary challenge that stands in the way is how do we know ωx(h). When one could build models to
(h) and then ωx(h) (e.g., using the replicator dynamics model as we introduced
predict the response
earlier), one could rework the above loss and apply standard gradient descent approaches. We provide
a concrete example and discussion in Appendix E. Without making any assumptions on the mapping
(h), one can only potentially rely on the bandit feedbacks from the decision subjects
between h and
D
to estimate the inﬂuence of h on
(h) - we also laid out a possibility in Appendix E too. It can also
be inferred from Eqn. (6) that the second challenge is the induced risk minimization might not even
be convex - due to the limit of space, we defer the detailed discussion again to the Appendix D.

D

D

9

012345K10−210−1ValueMaxLB012345K10−210−1ValueDiﬀUBUnder review as a conference paper at ICLR 2023

7 ETHICAL STATEMENT

The primary goal of our study is to put human in the center when considering domain shift. The
development of the paper is fully aware of any fairness concerns and we expect positive societal
impact. Unawareness of the potential distribution shift might lead to unintended consequence when
training a machine learning model. One goal of this paper is to raise awareness of this issue for a safe
deployment of machine learning methods in high-stake societal applications.

A subset of our results are developed under assumptions (e.g., Theorem 4.6). Therefore we want to
caution readers of potential misinterpretation of applicability of the reported theoretical guarantees.
Our contributions are mostly theoretical and our experiments use synthetic agent models to simulate
distribution shift. A future direction is to collect real human experiment data to support the ﬁndings.
Our paper ends with discussing the challenges in learning under the responding distribution and other
objectives that might arise.

We believe this is a promising research direction for the machine learning community, both as a
unaddressed technical problem and a stepstone for putting human in the center when training a
machine learning model.

8 REPRODUCIBILITY STATEMENT

We provide the following checklist for the purpose of reproducibility:

1. Generals:

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes] We have stated our assumptions
and limitations of the results. We also discussed the limitations in the conclusion.
(c) Did you discuss any potential negative societal impacts of your work? [Yes] One of our
work’s goals is to raise awareness of this issue for a safe deployment of machine learning
methods in high-stake societal applications. We discuss the potential misinterpretation
of our results in conclusion.

(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes]
(b) Did you include complete proofs of all theoretical results? [Yes] We present the

complete proofs in the appendix.

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-
mental results (either in the supplemental material or as a URL)? [Yes] We included
experiment details in the appendix and submitted the implementation in the supplemen-
tary materials.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)? [Yes]

(c) Did you report error bars (e.g., with respect to the random seed after running experi-
ments multiple times)? [N/A] In our controlled experiment, we do not tune parameters
and do not observe a signiﬁcant variance in the results.

(d) Did you include the total amount of compute and the type of resources used (e.g., type

of GPUs, internal cluster, or cloud provider)? [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]
(b) Did you mention the license of the assets? [Yes]
(c) Did you include any new assets either in the supplemental material or as a URL? [No]

10

Under review as a conference paper at ICLR 2023

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

REFERENCES

Syed Mumtaz Ali and Samuel D Silvey. A general class of coefﬁcients of divergence of one
distribution from another. Journal of the Royal Statistical Society: Series B (Methodological), 28
(1):131–142, 1966.

Kamyar Azizzadenesheli, Anqi Liu, Fanny Yang, and Animashree Anandkumar. Regularized learning

for domain adaptation under label shifts. arXiv preprint arXiv:1903.09734, 2019.

Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Vaughan.

A theory of learning from different domains. Machine Learning, 79:151–175, 2010.

Board of Governors of the Federal Reserve System (US). Report to the congress on credit scoring
and its effects on the availability and affordability of credit. Board of Governors of the Federal
Reserve System, 2007.

Anirban Chakraborty, Manaar Alam, Vishal Dey, Anupam Chattopadhyay, and Debdeep Mukhopad-

hyay. Adversarial attacks and defences: A survey, 2018.

Yatong Chen, Jialu Wang, and Yang Liu. Strategic recourse in linear classiﬁcation. arXiv preprint

arXiv:2011.00355, 2020a.

Yiling Chen, Yang Liu, and Chara Podimata. Learning strategy-aware linear classiﬁers, 2020b.

Koby Crammer, Michael Kearns, and Jennifer Wortman. Learning from multiple sources. Journal of

Machine Learning Research, 9(8), 2008.

Shai Ben David, Tyler Lu, Teresa Luu, and D´avid P´al. Impossibility theorems for domain adaptation.
In Proceedings of the Thirteenth International Conference on Artiﬁcial Intelligence and Statistics,
pp. 129–136. JMLR Workshop and Conference Proceedings, 2010.

Ofer Dekel, Felix Fischer, and Ariel D. Procaccia. Incentive compatible regression learning. J.

Comput. Syst. Sci., 76(8):759–777, December 2010.

Jinshuo Dong, Aaron Roth, Zachary Schutzman, Bo Waggoner, and Zhiwei Steven Wu. Strategic clas-
siﬁcation from revealed preferences. In Proceedings of the 2018 ACM Conference on Economics
and Computation, EC ’18, pp. 55–70, New York, NY, USA, 2018. Association for Computing
Machinery.

Abraham D Flaxman, Adam Tauman Kalai, and H Brendan McMahan. Online convex optimization

in the bandit setting: gradient descent without a gradient. arXiv preprint cs/0408007, 2004.

Daniel Friedman and Barry Sinervo. Evolutionary games in natural, social, and virtual worlds.

Oxford University Press, 2016.

Mingming Gong, Kun Zhang, Tongliang Liu, Dacheng Tao, Clark Glymour, and Bernhard Sch¨olkopf.
Domain adaptation with conditional transferable components. In International conference on
machine learning, pp. 2839–2848. PMLR, 2016.

11

Under review as a conference paper at ICLR 2023

Bryce Goodman and Seth Flaxman. European union regulations on algorithmic decision-making and

a “right to explanation”. AI Magazine, 38(3):50–57, Oct 2017.

Arthur Gretton, Alex Smola, Jiayuan Huang, Marcel Schmittfull, Karsten Borgwardt, and Bernhard
Sch¨olkopf. Covariate shift by kernel mean matching. Dataset shift in machine learning, 3(4):5,
2009.

Jiaxian Guo, Mingming Gong, Tongliang Liu, Kun Zhang, and Dacheng Tao. Ltf: A label transfor-
mation framework for correcting label shift. In International Conference on Machine Learning, pp.
3843–3853. PMLR, 2020.

Nika Haghtalab, Nicole Immorlica, Brendan Lucier, and Jack Z. Wang. Maximizing welfare
with incentive-aware evaluation mechanisms. In Christian Bessiere (ed.), Proceedings of the
Twenty-Ninth International Joint Conference on Artiﬁcial Intelligence, IJCAI-20, pp. 160–166.
International Joint Conferences on Artiﬁcial Intelligence Organization, 2020. doi: 10.24963/ijcai.
2020/23. URL https://doi.org/10.24963/ijcai.2020/23.

Moritz Hardt, Nimrod Megiddo, Christos Papadimitriou, and Mary Wootters. Strategic classiﬁcation.
In Proceedings of the 2016 ACM Conference on Innovations in Theoretical Computer Science, pp.
111–122, New York, NY, USA, 2016a. Association for Computing Machinery.

Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. In Advances

in Neural Information Processing Systems, pp. 3315–3323, 2016b.

Ling Huang, Anthony D Joseph, Blaine Nelson, Benjamin IP Rubinstein, and J Doug Tygar. Ad-
versarial machine learning. In ACM Workshop on Security and Artiﬁcial Intelligence, pp. 43–58,
2011.

Jing Jiang. A literature survey on domain adaptation of statistical classiﬁers. URL: http://sifaka. cs.

uiuc. edu/jiang4/domainadaptation/survey, 3:1–12, 2008.

Guoliang Kang, Lu Jiang, Yi Yang, and Alexander G Hauptmann. Contrastive adaptation network
for unsupervised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 4893–4902, 2019.

Jon Kleinberg and Manish Raghavan. How do classiﬁers induce agents to invest effort strategically?

ACM Transactions on Economics and Computation (TEAC), 8(4):1–23, 2020.

Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M. Hospedales. Learning to generalize: Meta-

learning for domain generalization, 2017.

Friedrich Liese and Igor Vajda. On divergences and informations in statistics and information theory.

IEEE Transactions on Information Theory, 52(10):4394–4412, 2006.

Zachary Lipton, Yu-Xiang Wang, and Alexander Smola. Detecting and correcting for label shift with
black box predictors. In International conference on machine learning, pp. 3122–3130. PMLR,
2018.

Yang Liu and Mingyan Liu. An online learning approach to improving the quality of crowd-sourcing.

ACM SIGMETRICS Performance Evaluation Review, 43(1):217–230, 2015.

Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I Jordan. Unsupervised domain adaptation

with residual transfer networks. arXiv preprint arXiv:1602.04433, 2016.

Daniel Lowd and Christopher Meek. Adversarial learning. In ACM SIGKDD International Conference

on Knowledge Discovery in Data Mining, pp. 641–647, 2005.

Celestine Mendler-D¨unner, Juan Perdomo, Tijana Zrnic, and Moritz Hardt. Stochastic optimization
for performative prediction. In Advances in Neural Information Processing Systems, pp. 4929–4939.
Curran Associates, Inc., 2020.

John Miller, Smitha Milli, and Moritz Hardt. Strategic classiﬁcation is causal modeling in disguise.

In International Conference on Machine Learning, pp. 6917–6926. PMLR, 2020.

12

Under review as a conference paper at ICLR 2023

Krikamol Muandet, David Balduzzi, and Bernhard Sch¨olkopf. Domain generalization via invariant

feature representation, 2013.

Zachary Nado, Shreyas Padhy, D. Sculley, Alexander D’Amour, Balaji Lakshminarayanan, and Jasper
Snoek. Evaluating prediction-time batch normalization for robustness under covariate shift, 2021.

Nicolas Papernot, Patrick McDaniel, and Ian Goodfellow. Transferability in machine learning: from

phenomena to black-box attacks using adversarial samples, 2016.

Juan Perdomo, Tijana Zrnic, Celestine Mendler-D¨unner, and Moritz Hardt. Performative prediction.

In International Conference on Machine Learning, pp. 7599–7609. PMLR, 2020.

Reilly Raab and Yang Liu. Unintended selection: Persistent qualiﬁcation rate disparities and

interventions. Advances in Neural Information Processing Systems, 34, 2021.

Andrew Selbst and Julia Powles. “meaningful information” and the right to explanation. In Sorelle A.
Friedler and Christo Wilson (eds.), Proceedings of the 1st Conference on Fairness, Accountability
and Transparency, volume 81 of Proceedings of Machine Learning Research, pp. 48–48. PMLR,
23–24 Feb 2018.

Yonadav Shavit, Benjamin Edelman, and Brian Axelrod. Causal strategic linear regression. In

International Conference on Machine Learning, pp. 8676–8686. PMLR, 2020.

Hidetoshi Shimodaira. Improving predictive inference under covariate shift by weighting the log-

likelihood function. Journal of statistical planning and inference, 90(2):227–244, 2000.

Chuanbiao Song, Kun He, Liwei Wang, and John E. Hopcroft. Improving the generalization of

adversarial training with domain adaptation, 2019.

Masashi Sugiyama, Matthias Krauledat, and Klaus-Robert M¨uller. Covariate shift adaptation by

importance weighted cross validation. Journal of Machine Learning Research, 8(5), 2007.

Masashi Sugiyama, Taiji Suzuki, Shinichi Nakajima, Hisashi Kashima, Paul von B¨unau, and Motoaki
Kawanabe. Direct importance estimation for covariate shift adaptation. Annals of the Institute of
Statistical Mathematics, 60(4):699–746, 2008.

Peter D. Taylor and Leo B. Jonker. Evolutionary stable strategies and game dynamics. Mathematical

Biosciences, 40(1):145–156, 1978. ISSN 0025-5564.

Karl Tuyls, Pieter Jan’T Hoen, and Bram Vanschoenwinkel. An evolutionary dynamical analysis
of multi-agent learning in iterated games. Autonomous Agents and Multi-Agent Systems, 12(1):
115–153, 2006.

Berk Ustun, Alexander Spangher, and Yang Liu. Actionable recourse in linear classiﬁcation. In
Proceedings of the Conference on Fairness, Accountability, and Transparency, pp. 10–19, 2019.

Thomas Varsavsky, Mauricio Orbes-Arteaga, Carole H. Sudre, Mark S. Graham, Parashkev Nachev,

and M. Jorge Cardoso. Test-time unsupervised domain adaptation, 2020.

Yevgeniy Vorobeychik and Murat Kantarcioglu. Adversarial Machine Learning. Morgan & Claypool

Publishers, 2018.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully

test-time adaptation by entropy minimization, 2021a.

Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun
Zeng, and Philip S. Yu. Generalizing to unseen domains: A survey on domain generalization,
2021b.

Qing Wang, S.R. Kulkarni, and S. Verdu. Divergence estimation of continuous distributions based
on data-dependent partitions. IEEE Transactions on Information Theory, 51(9):3064–3074, 2005.
doi: 10.1109/TIT.2005.853314.

Bianca Zadrozny. Learning and evaluating classiﬁers under sample selection bias. In Proceedings of

the twenty-ﬁrst international conference on Machine learning, pp. 114, 2004.

13

Under review as a conference paper at ICLR 2023

Kai Zhang, Vincent Zheng, Qiaojun Wang, James Kwok, Qiang Yang, and Ivan Marsic. Covariate
shift in hilbert space: A solution via surrogate kernels. In International Conference on Machine
Learning, pp. 388–395. PMLR, 2013a.

Kun Zhang, Bernhard Sch¨olkopf, Krikamol Muandet, and Zhikun Wang. Domain adaptation under
target and conditional shift. In International Conference on Machine Learning, pp. 819–827.
PMLR, 2013b.

Kun Zhang, Mingming Gong, Petar Stojanov, Biwei Huang, QINGSONG LIU, and Clark Glymour.
Domain adaptation as a problem of inference on graphical models. In H. Larochelle, M. Ranzato,
R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems,
volume 33, pp. 4965–4976. Curran Associates, Inc., 2020.

Yuchen Zhang, Tianle Liu, Mingsheng Long, and Michael Jordan. Bridging theory and algorithm for
domain adaptation. In International Conference on Machine Learning, pp. 7404–7413. PMLR,
2019.

14

Under review as a conference paper at ICLR 2023

A APPENDIX

We arrange the appendix as follows:

Appendix A.1 provides some real life scenarios where transparent models are useful or required.

Appendix A.2 provides comparisons of our setting and other sub-areas in domain adaptation.

Appendix A.3 provides proof for Theorem 3.1.

Appendix A.4 provides proof for Theorem 3.2.

Appendix A.5 provides proof of Theorem 3.3.

Appendix A.6 provides proof for Proposition 4.1.

Appendix A.7 provides proof for Theorem 4.2.

Appendix A.8 provides proof for Theorem 4.6.

Appendix A.9 provides omitted assumptions and proof for Section 4.3.

Appendix A.10 provides proof for Theorem 5.1.

Appendix A.11 provides proof for Theorem B.1.
Appendix A.12 provides proof for Proposition B.2.

Appendix B provides additional lower bound and examples for the target shift setting.

Appendix C provides missing experimental results , including new experimental results using
synthetic datasets generated according to causal graphs deﬁned in Figure 2. We also add additional
experimental results on credit score data set.

Appendix D discusses challenges in minimizing induced risk.

Appendix E provides discussions on how to directly minimize the induced risk.

Appendix F provides discussions on adding regularization to the objective function.

Appendix G provides discussions on the tightness of our theoretical bounds.

•

•

•

•

•

•

•

•

•

•

•

•

•

•

•

•

•

•

A.1 EXAMPLE USAGES OF TRANSPARENT MODELS

As we mentioned in Section 1, there is an increasing requirement of making the decision rule to be
transparent due to its potential consequences impacts to individual decision subject. Here we provide
the following reasons for using transparent models:

• Government regulation may require the model to be transparent, especially in public services;

• In some cases, companies may want to disclose their models so users will have explanations

and are incentivized to better use the provided services.

• Regardless of whether models are published voluntarily, model parameters can often be

inferred via well-known query “attacks”.

In addition, we name some concrete examples of some real-life applications:

• Consider the Medicaid health insurance program in the United States, which serves low-
income people. There is an obligation to provide transparency/disclose the rules (model
to automate the decisions) that decide whether individuals qualify for the program — in
fact, most public services have ”terms” that are usually set in stone and explained in the
documentation. Agents can observe the rules and will adapt their proﬁles to be qualiﬁed if
needed. For instance, an agent can decide to provide additional documentation they need to
guarantee approval. For more applications along these lines, please refer to this report5.

• Credit score companies directly publish their criteria for assessing credit risk scores. In loan
application settings, companies actually have the incentive to release criteria to incentivize
agents to meet their qualiﬁcations and use their services.Furthermore, making decision
models transparent will gain the trust of users.

5https://datasociety.net/library/poverty-lawgorithms/

15

Under review as a conference paper at ICLR 2023

• It is also known that it is possible to steal model parameters, if agents have incentives to do
so6. For instance, spammers frequently infer detection mechanisms by sending different
email variants; they then adjust their spam content accordingly.

A.2 COMPARISON OF OUR SETTING AND SOME AREAS IN DOMAIN ADAPTATION

We compare our setting (We address it as IDA, representing “induced domain adaptation”) with the
following areas:

• Adversarial attack Chakraborty et al. (2018); Papernot et al. (2016); Song et al. (2019): in
adversarial attack, the true label Y stays the same for the attacked feature, while in IDA,
we allow the true label to change as well. One can think of adversarial attack as a speciﬁc
form of IDA where the induced distribution has a speciﬁc target, that is to maximize the
classiﬁer’s error by only perturbing/modifying. Our transferability bound does, however,
provide insights for how standard training results transfer to the attack setting.

• Domain generalization Wang et al. (2021b); Li et al. (2017); Muandet et al. (2013): the
goal of domain generalization is to learn a model that can be generalized to any unseen
distribution; Similar to our setting, one of the biggest challenges in domain generalization
also the lack of target distribution during training. The major difference, however, is that our
focus is to understand how the performance of a classiﬁer trained on the source distribution
degrades when evaluated on the induced distribution (which depends on how the population
of decision subjects responds); this degradation depends on the classiﬁer itself.

• Test-time adaptation Varsavsky et al. (2020); Wang et al. (2021a); Nado et al. (2021): the
issue of test-time adaptation falls into the classical domain adaptation setting where the
adaptation is independent of the model being deployed. Applying this technique to solve
S(h)
our problem requires accessing data (either unsupervised or supervised) drawn from
for each h being evaluated during different training epochs.

D

A.3 PROOF OF THEOREM 3.1

Proof. We ﬁrst establish two lemmas that will be helpful for bounding the errors of a pair of classiﬁers.
Both are standard results from the domain adaption literature Ben-David et al. (2010).
Lemma A.1. For any hypotheses h, h(cid:48)

and distributions

(cid:48),

,

∈ H

ErrD(h, h(cid:48))

|

−

ErrD(cid:48)(h, h(cid:48))

| ≤

D

D
dH×H(
D
2

,

(cid:48))

.

D

Proof. Deﬁne the-cross prediction disagreement between two classiﬁers h, h(cid:48) on a distribution
ErrD(h, h(cid:48)) := PD(h(X)

as

D

= h(cid:48)(X)). By the deﬁnition of the
= h(cid:48)(X))
(cid:48)) = 2 sup

PD(h(X)

H−

divergence,
PD(cid:48)(h(X)

dH×H(

,

D

D

h,h(cid:48)∈H |

= h(cid:48)(X))
|

−
ErrD(cid:48)(h, h(cid:48))

|

= 2 sup

ErrD(h, h(cid:48))

h,h(cid:48)∈H |
ErrD(h, h(cid:48))
|

−

−
ErrD(cid:48)(h, h(cid:48))
|

.

2

≥

Another helpful lemma for us is the well-known fact that the 0-1 error obeys the triangle inequality
(see, e.g., Crammer et al. (2008)):

Lemma A.2. For any distribution
have ErrD(f1, f2)

D
ErrD(f1, f3) + ErrD(f2, f3).

≤

over instances and any labeling functions f1, f2, and f3, we

Denote by ¯h∗ the ideal joint hypothesis, which minimizes the combined error:

¯h∗ := arg min

h(cid:48)∈H

ErrD(h)(h(cid:48)) + ErrDS (h(cid:48))

6https://www.wired.com/2016/09/how-to-steal-an-ai/

16

(cid:54)
(cid:54)
(cid:54)
Under review as a conference paper at ICLR 2023

We have:

ErrD(h)(h)

≤

≤

≤

ErrD(h)(¯h∗) + ErrD(h)(h, ¯h∗)
ErrD(h)(¯h∗) + ErrDS (h, ¯h∗) + (cid:12)
ErrD(h)(¯h∗) + ErrDS (h) + ErrDS (¯h∗) +

−
dH×H(

1
2

(cid:12)ErrD(h)(h, ¯h∗)

ErrDS (h, ¯h∗)(cid:12)
(cid:12)

(Lemma A.2)

D

(h))

S,

D

(Lemma A.1)

(Deﬁnition of ¯h∗)

= ErrDS (h) + λDS→D(h) +

1
2

dH×H(

S,

D

D

(h)).

A.4 PROOF OF THEOREM 3.2

Proof. Invoking Theorem 3.1, and replacing h with h∗

T and S with

(h∗

T ), we have

ErrD(h)(h∗
T )

≤

Now observe that

ErrD(h∗

T )(h∗

T ) + λD(h)→D(h∗

T ) +

1
2

D
dH×H(

(h∗

T ),

(h))

D

D

(7)

ErrD(h)(h)

≤

≤

≤

≤

≤

ErrD(h)(h∗
ErrD(h)(h∗

T ) + ErrD(h)(h, h∗
T )
T )(h, h∗

T ) + ErrD(h∗

T ) +

ErrD(h)(h∗

T ) + ErrD(h∗

T )(h, h∗

T ) +

(cid:12)
(cid:12)ErrD(h)(h, h∗
(cid:12)
T )
1
(h∗
2

dH×H(

D

T ),

ErrD(h∗

(cid:12)
T )(h, h∗
(cid:12)
T )
(cid:12)

−

(h))

D

(by Lemma A.1)

ErrD(h)(h∗

T ) + ErrD(h∗

T )(h) + ErrD(h∗

T )(h∗

T ) +

1
2

dH×H(

D

(h∗

T ),

(h))

D

(by Lemma A.2)

ErrD(h∗

T )(h∗

T ) + λD(h)→D(h∗

T ) +

dH×H(

(h∗

T ),

(h))

D

(by equation 7)

D

+ ErrD(h∗

T )(h) + ErrD(h∗

T )(h∗

T ) +

dH×H(

D

(h∗

T ),

(h))

D

Adding ErrD(h)(h) to both sides and rearranging terms yields

1
2
1
2

2ErrD(h)(h)

2ErrD(h∗

T )(h∗
T )

−

ErrD(h)(h) + ErrD(h∗

T )(h) + λD(h)→D(h∗

≤
= ΛD(h)→D(h∗

T )(h) + λD(h)→D(h∗

T ) + dH×H(

T ) + dH×H(
(h∗
T ),

T ),

(h∗
D
(h))

D

D

(h))

D

Dividing both sides by 2 completes the proof.

A.5 PROOF OF THEOREM 3.3

Proof. Using the triangle inequality of dTV, we have

D

dTV(

dTV(

Y (h))

DY |S,

Dh|S) + dTV(
≤
and by the deﬁnition of dTV, the divergence term dTV(
DY |S,

DY |S,

dTV(

h(h),

Y (h))

D

(8)

h(h)) + dTV(

Dh|S,
D
D
Y (h)) becomes
DY |S,
D
PDS (h(x) = +1)
−
EDS [h(X)] + 1
2
(cid:12)
(cid:12)
(cid:12)
(cid:12)

−
EDS [h(X)]
2

(cid:12)
(cid:12)
(cid:12)
(cid:12)

|

h(X)

]
|

Dh|S) =
=

PDS (Y = +1)
|
(cid:12)
EDS [Y ] + 1
(cid:12)
(cid:12)
2
(cid:12)
(cid:12)
EDS [Y ]
(cid:12)
(cid:12)
2
(cid:12)
1
2 ·

EDS [

−

−

Y
|
≤
= ErrDS (h)

=

Similarly, we have

dTV(

h(h),

D

Y (h))

D

≤

ErrD(h)(h)

17

Under review as a conference paper at ICLR 2023

As a result, we have

ErrDS (h) + ErrD(h)(h)

dTV(
dTV(

DY |S,
DY |S,

≥

≥

which implies

Dh|S) + dTV(
h(h),
D
D
Dh|S,
dTV(
Y (h))
D

−

D

Y (h))
h(h))

(by equation 8)

max
{

ErrDS (h), ErrD(h)(h)

} ≥

dTV(

DY |S,

Y (h))

D

−
2

dTV(

Dh|S,

h(h))

D

.

A.6 PROOF OF PROPOSITION 4.1

Proof.

ED(h)[(cid:96)(h; X, Y )]
(cid:90)

PD(h)(X = x, Y = y)(cid:96)(h; x, y) dxdy

=

=

=

=

(cid:90)

(cid:90)

(cid:90)

PDS (Y = y

PDS (Y = y

PDS (Y = y

X = x)
|

X = x)
|

X = x)
|

·

·

·

PD(h)(X = x)(cid:96)(h; x, y) dxdy

PDS (X = x)

PDS (X = x)

·

·

PD(h)(X = x)
PDS (X = x) ·

(cid:96)(h; x, y) dxdy

ωx(h)

·

(cid:96)(h; x, y) dxdy

=EDS [ωx(h)

(cid:96)(h; x, y)]

·

A.7 PROOF OF THEOREM 4.2

S. Let the average importance weight induced by h∗

S be

Proof. We start from the error induced by h∗
¯ω(h∗

S) = EDS [ωx(h∗
S )(h∗
ErrD(h∗

S)]; we add and subtract this from the error:
S(x)
S(x)

= y)]
= y)] + EDS [(ωx(h∗
S)

S) = EDS [ωx(h∗
S)
= EDS [¯ω(h∗
S)

1(h∗
·
1(h∗

In fact, ¯ω(h∗

S) = 1, since

·

(cid:90)

¯ω(h∗

S))

·

−

1(h∗

S(x)

= y)]

¯ω(h∗

S) =EDS [ωx(h∗

S)] =
(cid:90) PD(h)(X = x)
PDS (X = x)

=

ωx(h∗

S)PDS (X = x)dx
(cid:90)

PDS (X = x)dx =

PD(h)(X = x)dx = 1

Now consider any other classiﬁer h. We have

S )(h∗
ErrD(h∗
S)
= EDS [1(h∗
S(x)
EDS [1(h(x)

≤
= EDS [¯ω(h)

·
= EDS [ωx(h)

= y)] + EDS [(ωx(h∗
S)
= y)] + EDS [(ωx(h∗
S)

S))

¯ω(h∗
−
¯ω(h∗
S))

1(h∗
·
1(h∗

S(x)
S(x)

= y)]

= y)]

−

·

1(h(x)

= y)] + EDS [(ωx(h∗
S)

¯ω(h∗

S))

·

−

1(h∗

(by optimality of h∗
S(x)

= y)]

S on

S)

D

(multiply by ¯ω(h∗

S) = 1)

1(h(x)

= y)] + EDS [(¯ω(h)

·

ωx(h))

·

−

1(h(x)

= y)]
(add and subtract ¯ω(h∗

S))

+ EDS [(ωx(h∗
S)
= ErrD(h)(h) + Cov(ωx(h∗

−

S))
S), 1(h∗

1(h∗
S(x)

¯ω(h∗

·

S(x)

= y)]

Cov(ωx(h), 1(h(x)

= y))

= y))

−

18

(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
(cid:54)
Under review as a conference paper at ICLR 2023

Moving the error terms to one side, we have

Cov(ωx(h), 1(h(x)

S )(h∗
ErrD(h∗
S)
−
S), 1(h∗
Cov(ωx(h∗
(cid:113)

Var(ωx(h∗

S))

(cid:112)

+

Var(ωx(h))

ErrD(h)(h)
S(x)
= y))
Var(1(h∗

−
S(x)
Var(1(h(x)

·

·
ErrS(h∗

S)(1

Var(ωx(h∗

S))

·

−
(cid:112)

= y))

= y))

Var(ωx(h∗

S))

ErrS(h∗

S) +

Var(ωx(h))

·
(cid:18)(cid:113)

ErrDS (h)

·

Var(ωx(h∗

S)) +

Var(ωx(h))

(cid:112)

≤

≤

=

≤

≤

(cid:113)

(cid:113)

(cid:112)

·

ErrDS (h)
(cid:19)

= y))

(

Cov(X, Y )
|

| ≤

(cid:112)Var(X)

Var(Y ))

·

ErrS(h∗

S)) +

(cid:112)

Var(ωx(h))

ErrDS (h)(1

·

−
(1

ErrDS (h))

ErrDS (h)

1)

≤

−

Since this holds for any h, it certainly holds for h = h∗
T .

A.8 OMITTED ASSUMPTIONS AND PROOF OF THEOREM 4.6

Denote X+(h) =

x : ωx(h)
{

1

}

≥

(cid:90)

and X−(h) =

x : ωx(h) < 1
. First we observe that
}
{

PDS (X = x)(1

PDS (X = x)(1

−

−

ωx(h))dx

ωx(h))dx = 0

X+(h)

(cid:90)

X−(h)

+

This is simply because of (cid:82)

x

PDS (X = x)

ωx(h)dx = (cid:82)

x

·

PD(h)(X = x)dx = 1.

Proof. Notice that in the setting of binary classiﬁcation, we can write the total variation distance
Y (h) as the difference between the probability of Y = +1 and the probability
between
of Y =

D

DY |S and
1:
−

Y (h))

D

DY |S,
dTV(
= (cid:12)
(cid:12)PDS (Y = +1)
(cid:12)
(cid:90)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:90)
(cid:12)
(cid:12)
(cid:12)

−
PDS (Y = +1
|

PDS (Y = +1
|

=

=

PD(h)(Y = +1)(cid:12)
(cid:12)

X = x)PDS (X = x)dx

−

(cid:90)

X = x)PDS (X = x)

(1

·

−

(cid:12)
(cid:12)
ωx(h))dx
(cid:12)
(cid:12)

X = x)PDS (X = x)ωx(h)dx
PDS (Y = +1
|

(cid:12)
(cid:12)
(cid:12)
(cid:12)

(9)

Similarly we have

dTV(

Dh|S,

h(h)) =

D

(cid:90)

(cid:12)
(cid:12)
(cid:12)
(cid:12)

PDS (h(x) = +1

X = x)PDS (X = x)
|

·

(1

−

(cid:12)
(cid:12)
ωx(h))dx
(cid:12)
(cid:12)

(10)

19

(cid:54)
(cid:54)
(cid:54)
(cid:54)
=

=

dTV(
(cid:12)
(cid:90)
(cid:12)
(cid:12)
(cid:12)
(cid:90)
(cid:12)
(cid:12)
(cid:12)

(cid:124)

+

(cid:90)

X−(h)

(cid:124)
(cid:90)

X+(h)

(cid:90)

X−(h)

X+(h)
(cid:90)

X−(h)

=

−

−
(cid:90)

=

+

(cid:90)

=

Under review as a conference paper at ICLR 2023

We can further expand the total variation distance between

DY |S and

Y (h) as follows:

D

Y (h))

DY |S,
D
X = x)PDS (X = x)
PDS (Y = +1
|

(cid:12)
(cid:12)
ωx(h))dx
(cid:12)
(cid:12)

(1

·

−

X = x)PDS (X = x)
PD(Y = +1
|

·

(1

−

ωx(h))dx

X+(h)

(cid:123)(cid:122)
≤0

PDS (Y = +1
|

X = x)PDS (X = x)

(cid:123)(cid:122)
>0

X = x)PDS (X = x)
PDS (Y = +1
|

PDS (Y = +1
|

X = x)PDS (X = x)

·

·

·

(cid:125)

(1

−

(cid:12)
(cid:12)
ωx(h))dx
(cid:12)
(cid:125)

ωx(h))dx

ωx(h))dx

(by Assumption 4.3)

(1

(1

−

−

PDS (Y = +1
X = x)PDS (X = x)
|

·

(ωx(h)

1)dx

−

PDS (Y = +1
X = x)PDS (X = x)
|

·

(ωx(h)

1)dx

−

(by equation 9)

X = x)PDS (X = x)
PDS (Y = +1
|

·

(ωx(h)

1)dx

−

Similarly, by assumption 4.4 and equation equation 10, we have

dTV(

Dh|S,

h(h)) =

D

(cid:90)

PDS (h(x) = +1

X = x)PDS (X = x)
|

·

(ωx(h)

1)dx

−

Thus we can bound the difference between dTV(

DY |S,

Y (h)) and dTV(

D

Dh|S,

h(h)) as follows:

D

dTV(
(cid:90)

D

dTV(

Y (h))

DY |S,
Dh|S,
X = x)PDS (X = x)
PDS (Y = +1
|

−

D

h(h))

(ωx(h)

1)dx

−

·

X = x)PDS (X = x)
PD(h(x) = +1
|

(cid:90)

=

(cid:90)

=

−
[PDS (Y = +1
X = x)
|

−
X = x)

= EDS [(PDS (Y = +1
|

> EDS [PDS (Y = +1
X = x)
|
= 0

PDS (h(x) = +1

(ωx(h)

1)dx

·
−
X = x)]PDS (X = x)
|

PDS (h(x) = +1

X = x)) (ωx(h)

−
|
−
X = x)]EDS [ωx(h)
PDS (h(x) = +1
|

−

1)dx

−

(ωx(h)

·
1)]

(by Assumption 4.5)

1]

−

Combining the above with Theorem 3.3, we have

max

ErrDS (h), ErrD(h)(h)
{

} ≥

dTV(

DY |S,

Y (h))

D

−
2

dTV(

Dh|S,

h(h))

D

> 0

A.9 OMITTED DETAILS FOR SECTION 4.3

With Setup 2 - Setup 4, we can further specify the important weight wx(h) for the strategic response
setting:

20

Under review as a conference paper at ICLR 2023

Lemma A.3. Recall the deﬁnition for the covariate shift important weight coefﬁcient ωx(h) :=
PD(h)(X=x)
PDS (X=x) , for our strategic response setting, we have,

wx(h) =






1,
τh−x
B ,
1
B (
1,

x
x
x + τh + 2B), x
x

−

B)
[0, τh
−
[τh
B, τh)
[τh, τh + B)
[τh + B, 1]

−

∈
∈
∈
∈

(11)

Proof for Lemma A.3:

Proof. We discuss the induced distribution

(h) by cases:

D

• For the features distributed between [0, τh

B]: since we assume the agents are rational,
−
B] will not perform any
under assumption 2, agents with feature that is smaller than [0, τh
kinds of adaptations, and no other agents will adapt their features to this range of features
either, so the distribution between [0, τh

B] will remain the same as before.

−

−
B, τh] can be directly calculated from assumption

• For the target distribution between [τh

3.

−

• For distribution between [τh, τh + B], consider a particular feature x(cid:63)

Setup 4, we know its new distribution becomes:

[τh, τh + B], under

∈

PD(h)(x = x(cid:63)) = 1 +

= 1 +

(cid:90) τh

x(cid:63)−B

(cid:90) τh

x(cid:63)−B

1
B

1
B

−
−
dz

τh−z
B
τh + z

dz

=

1
B

(

−

x(cid:63) + τh + 2B)

• For the target distribution between [τh + B, 1]: under assumption 2 and 4, we know that no
agents will change their feature to this feature region. So the distribution between [τh + B, 1]
remains the same as the source distribution.

Recall the deﬁnition for the covariate shift important weight coefﬁcient ωx(h) :=
distribution of ωx(h) after agents’ strategic responding becomes:

PD(h)(X=x)
PDS (X=x) , the

x
x
x + τh + 2B), x

B) and x

[0, τh
−
B, τh)
[τh
[τh, τh + B)

−

[τh + B, 1]

∈

(12)

∈
∈
∈

otherwise

1,
τh−x
B ,
1
B (
0,

−

ωx(h) =






Proof for Proposition 4.7:

Proof. According to Lemma A.3, we can compute the variance of wx(h) as Var(wx(h)) =
E(wx(h)2)
3 B. Then by plugging it to the general bound for Theorem 4.2 gives us
the desirable result.

E(wx(h)2) = 2

−

21

Under review as a conference paper at ICLR 2023

A.10 PROOF OF THEOREM 5.1

Proof. Deﬁning p := PDS (Y = +1), p(h) = PD(h)(Y = +1), we have

ErrD(h∗

S )(h∗

S) = p(h∗
S)

Err+(h∗

S) + (1

p(h∗

S))

·

−

Err−(h∗
S)

·

= p
(cid:124)

·

Err+(h∗

S) + (1
(cid:123)(cid:122)
(I)

p)

·

−

We can expand (I) as follows:

(by deﬁnitions of p(h∗
+(p(h∗
S)

p)[Err+(h∗
S)

S), Err+(h∗

S), and Err−(h∗
Err−(h∗

S))
(13)

S)]

−

−

Err−(h∗
S)
(cid:125)

S) + (1
T ) + (1

Err+(h∗
p
·
Err+(h∗
p
≤
·
= p(h∗
Err+(h∗
T )
·
T )(h∗
= ErrD(h∗

T ) + (p

p)
−
p)
−
T ) + (1

Err−(h∗
S)
Err−(h∗
T )
Err−(h∗
p(h∗
T ))
·
[Err+(h∗
T )
T ))

·
·
−
p(h∗

−

·

T ) + (p

p(h∗
−
Err−(h∗
T )] .

T ))

−

(by optimality of h∗
Err−(h∗
[Err+(h∗
T )

S on
T )]

−

·

S)

D

Plugging this back into equation 13, we have

ErrD(h∗

S )(h∗
S)
Notice that

−

ErrD(h∗

T )(h∗
T )

(p(h∗
S)

−

≤

p)[Err+(h∗
S)

−

Err−(h∗

S)] + (p

p(h∗

T ))

·

−

[Err+(h∗
T )

−

Err−(h∗

T )]

0.5(Err+(h)

−

Err−(h)) = 0.5
= 0.5

P(h(X) = +1

0.5

1
−
PDu(h(X) = +1)

·

·
−

Y = +1)
|

−

0.5

·

P(h(X) = +1

Y =
|

−

1)

where

D

u is a distribution with uniform prior. Then

(p(h∗
p)[Err+(h∗
S)
S)
−
T ))[Err+(h∗
p(h∗
(p
T )

−

−
−

Err−(h∗
Err−(h∗

S)] = 2(p(h∗
S)
−
p(h∗
T )] = 2(p

p)
T ))

−

(0.5
(0.5

·
·

−
−

PDu(h(X) = +1))
PDu (h(X) = +1))

Adding together these two equations yields

(0.5

(p(h∗
S)
−
= 2(p(h∗
S)
= (p(h∗
S)
+ 2p
p(h∗
S)
+ 2p

−
PDu (h∗

p)[Err+(h∗
S)
p)
−
·
p(h∗
T ))
−
(PDu (h∗
·
p(h∗
T )
| ·
−
PDu (h∗
S(X) = +1)
· |

−
2 (p(h∗
−
S(X) = +1)

(1 + 2

≤ |

|

Err−(h∗

S)] + (p

p(h∗
S(X) = +1)) + 2(p
S(X) = +1)

S)PDu (h∗

−

T ))

[Err+(h∗
T )
·
−
p(h∗
(0.5
T ))
−
·
T )PDu (h∗
p(h∗
−

Err−(h∗

T )]
PDu (h∗
T (X) = +1))

−

−
PDu (h∗

T (X) = +1))

PDu (h∗
S(X) = +1)
PDu (h∗

−
T (X) = +1)
|

−

PDu (h∗

T (X) = +1)

)
|

T (X) = +1))

Meanwhile,

PDu (h∗
|
0.5

S(X) = +1)

PD|Y =+1(h∗

PDu (h∗
S(X) = +1)

−

T (X) = +1)
|

PD|Y =+1(h∗

≤

· |
+ 0.5
· |
= 0.5 (dTV(

PD|Y =−1(h∗
+(h∗
S),

S(X) = +1)
T )) + dTV(

+(h∗

PD|Y =−1(h∗
−(h∗
−(h∗
S),

T (X) = +1)
|
T ))

D

D

D

−

−

D

T (X) = +1)
|

Combining equation 14 and equation 15 gives

|

· |

(1 + 2

PDu (h∗

p(h∗
p(h∗
T )
S)
| ·
−
PDu (h∗
S(X) = +1)
+ 2p
· |
p(h∗
p(h∗
(1 + dTV(
T )
S)
−
S),
(dTV(
+ p
D
·
p(h∗
p(h∗
+ (1 + p)
T )
S)
|
−

S(X) = +1)
PDu (h∗
−
+(h∗
S),
D
D
+(h∗
T )) + dTV(
D
+(h∗
S),
(dTV(

| ·
+(h∗

−(h∗

D

D

D

·

≤ |

≤ |

−

T (X) = +1)
|
+(h∗

T )) + dTV(

−(h∗
D
−(h∗
S),
T ))
D
+(h∗
T )) + dTV(

PDu (h∗

T (X) = +1)

)
|

S),

−(h∗

T ))

D

−(h∗

S),

−(h∗

T )) .

D

D

22

(14)

(15)

Under review as a conference paper at ICLR 2023

A.11 PROOF OF THEOREM B.1

We will make use of the following fact:
Lemma A.4. Under label shift, TPRS(h) = TPRh(h) and FPRS(h) = FPRh(h).

Proof. We have

TPRh(h) =PD(h)(h(X) = +1

Y = +1)
|

(cid:90)

(cid:90)

(cid:90)

(cid:90)

(cid:90)

=

=

=

=

=

PD(h)(h(X) = +1, X = x

Y = +1)dx
|

X = x, Y = +1)PD(h)(X = x
PD(h)(h(X) = +1
|

Y = +1)dx
|

1(h(x) = +1)PD(h)(X = x

Y = +1)dx

|

1(h(x) = +1)PDS (X = x

Y = +1)dx
|

(by deﬁnition of label shift)

PDS (h(X) = +1

|

X = x, Y = +1)PDS (X = x

Y = +1)dx
|

=TPRS(h)

The argument for TPRh(h) = TPRS(h) is analogous.

Now we proceed to prove the theorem.

Proof of Theorem B.1. In section 3.2 we showed a general lower bound on the maximum of ErrDS (h)
and ErrD(h)(h):

max
{

ErrDS (h), ErrD(h)(h)

} ≥

dTV(

DY |S,

Y (h))

D

dTV(

Dh|S,

h(h))

D

−
2

In the case of label shift, and by the deﬁnitions of p and p(h),

dTV(

DY |S,

D

Y (h)) =

|

PDS (Y = +1)

PD(h)(Y = +1)
|

=

p
|

−

−

p(h)

|

In addition, we have

Dh|S = PS(h(X) = +1) = p

·

TPRS(h) + (1

p)

·

−

FPRS(h)

(16)

(17)

Similarly

Therefore

D

h(h) = PD(h)(h(X) = +1)
TPRh(h) + (1
TPRS(h) + (1

= p(h)
= p(h)

·
·

p(h))
p(h))

·
·

−
−

FPRh(h)
FPRS(h)

(by Lemma A.4)

(18)

dTV(

Dh|S,

D

PDS (h(X) = +1)
h(h)) =
|
p(h))
(p
=
|

−

·

−

PD(h)(h(X) = +1)
|
FPRS(h)

p)

TPRS(h) + (p(h)

−

·

|

(By equation 18 and equation 17)

=

p
|

−

p(h)

| · |

TPRS(h)

−

FPRS(h)

|

(19)

which yields:

dTV(

DY |S,

D

Y (h))

dTV(

Dh|S,

−

h(h)) =

D

p

|

−

p(h)

(1
|

completing the proof.

23

TPRS(h)

− |
(By equation 16 and equation 19)

−

FPRS(h)
|

)

Under review as a conference paper at ICLR 2023

A.12 PROOF OF PROPOSITION B.2

Proof.

p(h∗
S)
|

(1

= |

−

p(h∗
T )

−
ErrDS (h∗
(1

ErrDS (h∗
S)
|
(1

≤

−
−

1
PDS (Y = +1)
| ·
S))TPRS(h∗
ErrDS (h∗
(1
S)
−
−
ErrDS (h∗
ErrDS (h∗
(1
S))
·
−
−
TPRS(h∗
ErrDS (h∗
S)
T )
−
ErrDS (h∗
ErrDS (h∗
(1
S))

| · |
·

−

T ))TPRS(h∗
T )
|
T ))
TPRS(h∗
T ))

T )
|

(20)

The inequality above is due to Lemma 7 of Liu & Liu (2015).

B LOWER BOUND AND EXAMPLE FOR TARGET SHIFT

B.1 LOWER BOUND

Now we discuss lower bounds. Denote by TPRS(h) and FPRS(h) the true positive and false positive
rates of h on the source distribution
Theorem B.1. Under target shift, any model h must incur the following error on either the

S. We prove the following:

D

S or

D

(h):

D

≥

ErrDS (h), ErrD(h)(h)
max
{
}
TPRS(h)
p(h)
p
|

− |

(1

−

| ·

−

2

FPRS(h)

)
|

.

D

Dh|S, and

Y (h)),
The proof extends the bound of Theorem 3.3 by further explicating each of dTV(
dTV(
< 0 unless
h(h)) under the assumption of target shift. Since
we have a trivial classiﬁer that has either TPRS(h) = 1, FPRS(h) = 0 or TPRS(h) = 0, FPRS(h) =
1, the lower bound is strictly positive. Taking a closer look, the lower bound is determined linearly
p(h). The difference is further determined by the
by how much the label distribution shifts: p
performance of h on the source distribution through 1
FPRS(h)
. For instance, when
|
TPRS(h) > FPRS(h), the quality becomes FNRS(h) + FPRS(h), that is the more error h makes,
the larger the lower bound will be.

DY |S,
|

TPRS(h)
|

TPRS(h)

FPRS(h)

− |

−

−

−

D

B.2 EXAMPLE USING REPLICATOR DYNAMICS

Let us instantiate the discussion using a speciﬁc ﬁtness function for the replicator dynamics model
(Section 2.1), which is the prediction accuracy of h for class +1:
[Fitness of Y = +1] := PDS (h(X) = +1

Y = +1)

(21)

|

Then we have E [Fitness of Y ] = ErrDS (h), and

p(h)
PDS (Y = +1)

=

PDS (h(X) = +1
Y = +1)
|
ErrDS (h)

Plugging the result back to our Theorem 5.1 we have
Proposition B.2. Under the replicator dynamics model in Eqn. (21),
as:

p(h∗
S)
|

p(h∗

T )
|

−

further bounds

PDS (Y = +1)

p(h∗
p(h∗
S)
T )
|
−
ErrDS (h∗
S)
|

−

·

| ≤
ErrDS (h∗
T )
ErrDS (h∗
S)

TPRS(h∗
S)
ErrDS (h∗
T )

| · |
·

TPRS(h∗

T )
|

.

−

That is, the difference between ErrD(h∗
between the two classiﬁers’ performances on the source data
evaluate the possible error transferability using the source data only.

S) and ErrD(h∗

D

T )(h∗

S )(h∗

T ) is further dependent on the difference
S. This offers an opportunity to

24

Under review as a conference paper at ICLR 2023

C MISSING EXPERIMENTAL DETAILS

C.1 SYNTHETIC EXPERIMENTS USING DAG

Synthetic experiments using simulated data We generate synthetic data sets from structural
equation models described on simple causal DAG in Figure 2 for covariate shift and target shift. To
Rd,
generate the induced distribution
, its induced features are precisely x(cid:48) = ∆(x, h).
so that when an input x encounters classiﬁer h
We provide details of the data generation processes and adaptation functions in Appendix C.

(h), we posit a speciﬁc adaptation function ∆ : Rd

× H →

∈ H

D

{

x1, . . . , xn

1. To compute h∗

and learn a “base” logistic regression model h(x) = σ(w
We take our training data set
·
}
x)7. We then consider the hypothesis class
x) >
[0, 1]
hτ
{
τ ]
S, the model that performs best on the source distribution, we simply vary τ
and take the hτ with lowest prediction error. Then, we posit a speciﬁc adaptation function ∆(x, hτ ).
Finally, to compute h∗
T , we vary τ from 0 to 1 and ﬁnd the classiﬁer hτ that minimizes the prediction
error on its induced data set

, where hτ (x) := 2
}

. We report our results in Figure 4.

1[σ(w

:=

H

−

∈

τ

·

|

·

∆(x1, hτ ), . . . , ∆(xn, hτ )
}

{

Figure 4: Results for synthetic experiments on simulated and real-world data. Diff := ErrD(h∗
T )(h∗
ErrDS (h∗
T ), ErrD(h∗
ErrD(h∗
T )
{
}
rem 4.2, and LB := lower bound speciﬁed in Theorem 4.6.

−
, UB := upper bound speciﬁed in Theo-

T ), Max := max

T )(h∗

S )(h∗
S)

Covariate Shift We specify the causal DAG for covariate shift setting in the following way:
X1
Unif(
(0, σ2
X2
1.2X1 +
2)
N
(0, σ2
X 2
X3
3)
1 +
Y := 2sign(X2 > 0)

1, 1)

∼ −

N

−

∼

∼

1

2 and σ2

where σ2
3 are parameters of our choices.
Adaptation function We assume the new distribution of feature X (cid:48)
way:

−

1 will be generated in the following

X (cid:48)

1 = ∆(X) = X1 + c

(h(X)

1)

∈

·
R1 > 0 is the parameter controlling how much the prediction h(X) affect the generating
where c
of X (cid:48)
1, namely the magnitude of distribution shift. Intuitively, this adaptation function means that if a
feature x is predicted to be positive (h(x) = +1), then decision subjects are more likely to adapt to
that feature in the induced distribution; Otherwise, decision subjects are more likely to be moving
away from x since they know it will lead to a negative prediction.

−

Target Shift We specify the causal DAG for target shift setting in the following way:

(Y + 1)/2
X1

Y = y

|

Bernoulli(α)
∼
∼ N[0,1](µy, σ2)
(0, σ2
0.8X1 +
2)
N
(0, σ2
3)

X2 =
−
X3 = 0.2Y +

N

7σ(
·

) is the logistic function and w

∈

R3 denotes the weights.

25

Under review as a conference paper at ICLR 2023

N[0,1] represents a truncated Gaussian distribution taken value between 0 and 1. α, µy, σ2,σ2
where
and σ2
3 are parameters of our choices.
Adaptation function We assume the new distribution of the qualiﬁcation Y (cid:48) will be updated in the
following way:

2

P(Y (cid:48) = +1
h(X) = h, Y = y) = chy, where
|

h, y
{

} ∈ {−

1, +1
}

where 0
and get predicted as h(X) = h to be qualiﬁed in the next step (Y (cid:48) = +1).

1 represents the likelihood for a person with original qualiﬁcation Y = y

chy

≤

≤

∈

R1

T )(h∗

T ), indicating the suboptimality of training on

Discussion of the Results For all four datasets, we do observe positive gaps ErrD(h∗
−
ErrD(h∗
S. The gaps are well bounded by the
theoretical results. For lower bound, the empirical observation and the theoretical bounds are roughly
within the same magnitude except for one target shift dataset, indicating the effectiveness of our
theoretical result. For upper bound, for target shift, the empirical observations are well within the
same magnitude of the theoretical bounds while the results for the covariate shift are relatively loose.

D

S )(h∗
S)

C.2 SYNTHETIC EXPERIMENTS USING REAL-WORLD DATA

On the preprocessed FICO credit score data set (Board of Governors of the Federal Reserve System
(US), 2007; Hardt et al., 2016b), we convert the cumulative distribution function (CDF) of TransRisk
score among demographic groups (denoted as A, including Black, Asian, Hispanic, and White)
into group-dependent densities of the credit score. We then generate a balanced sample where each
group has equal representation, with credit scores (denoted as Q) initialized by sampling from the
corresponding group-dependent density. The value of attributes for each data point is then updated
under a speciﬁed dynamics (detailed in Appendix C.2.1) to model the real-world scenario of repeated
resource allocation (with decision denoted as D).

C.2.1 PARAMETERS FOR DYNAMICS

Since we are considering the dynamic setting, we further specify the data generating process in the
following way (from time step T = t to T = t + 1):

∼
∼

(cid:15)1, (cid:15)1]
(cid:15)2, (cid:15)2]

1.5Qt + U [
0.8At + U [
At +
Bernoulli(qt) for a given value of Qt = qt

Xt,1
Xt,2
Xt,3
Yt
Dt = ft(At, Xt,1, Xt,2, Xt,3)

−
−
(0, σ2)

∼
∼

N

Qt+1 =
·
At+1 = At (ﬁxed population)

Qt
{

[1 + αD(Dt) + αY (Yt)]

}(0,1]

{·}(0,1] represents truncated value between the interval (0, 1], ft(
·

where
) represents the decision
policy from input features, and (cid:15)1, (cid:15)2, σ are parameters of choices. In our experiments, we set
(cid:15)1 = (cid:15)2 = σ = 0.1.

Within the same time step, i.e., for variables that share the subscript t, Qt and At are root causes for
all other variables (Xt,1, Xt,2, Xt,3, Dt, Yt). At each time step T = t, the institution ﬁrst estimates
the credit score Qt (which is not directly visible to the institution, but is reﬂected in the visible
outcome label Yt) based on (At, Xt,1, Xt,2, Xt,3), then produces the binary decision Dt according
to the optimal threshold (in terms of the accuracy).

For different time steps, e.g., from T = t to T = t + 1, the new distribution at T = t + 1 is induced by
the deployment of the decision policy Dt. Such impact is modeled by a multiplicative update in Qt+1
from Qt with parameters (or functions) αD(
) that depend on Dt and Yt, respectively. In
) and αY (
·
·
our experiments, we set αD = 0.01 and αY = 0.005 to capture the scenario where one-step inﬂuence
of the decision on the credit score is stronger than that for ground truth label.

26

Under review as a conference paper at ICLR 2023

(a) L1 penalty, strong regularization strength.

(b) L1 penalty, strong regularization strength.

(c) L1 penalty, medium regularization strength.

(d) L1 penalty, medium regularization strength.

(e) L1 penalty, weak regularization strength.

(f) L1 penalty, weak regularization strength.

T ), ErrD(h∗

ErrDS (h∗
{

Figure 5: Results of applying L1 penalty with different strength when constructing h∗
S.
:=
and LB := lower bound speciﬁed in Theorem 4.6. The

The left column consisting of panels (a),
max
right column consisting of panels (b), (d), and (f) compares Diff := ErrD(h∗
−
T ) and UB := upper bound speciﬁed in Theorem 4.2. For each time step
ErrD(h∗
K = k, we compute and deploy the source optimal classiﬁer h∗
S and update the credit
score for each individual according to the received decision as the new reality for time step
K = k + 1.

(c), and (e) compares Max

T )(h∗
T )
}

S )(h∗
S)

T )(h∗

C.2.2 ADDITIONAL EXPERIMENTAL RESULTS

In this section, we present additional experimental results on the real-world FICO credit score data
set. With the initialization of the distribution of credit score Q and the speciﬁed dynamics, we present
results comparing the inﬂuence of vanilla regularization terms in decision-making (when estimating
the credit score Q) on the calculation of bounds for induced risks.8 In particular, we consider L1
norm (Figure 5) and L2 norm (Figure 6) regularization terms when optimizing decision-making
policies on the source domain. As we can see from the results, applying vanilla regularization terms
(e.g., L1 norm and L2 norm) on source domain without speciﬁc considerations of the inducing-risk
mechanism does not provide signiﬁcant performance improvement in terms of smaller induced risk.
For example, there is no signiﬁcant decrease of the term Diff as the regularization strength increases,
for both L1 norm (Figure 5) and L2 norm (Figure 6) regularization terms.

8The regularization that involves induced risk considerations will be discussed in Appendix F.

27

012345K10−210−1ValueMaxLB012345K10−210−1ValueDiﬀUB012345K10−210−1ValueMaxLB012345K10−210−1ValueDiﬀUB012345K10−210−1ValueMaxLB012345K10−210−1ValueDiﬀUBUnder review as a conference paper at ICLR 2023

(a) L2 penalty, strong regularization strength.

(b) L2 penalty, strong regularization strength.

(c) L2 penalty, medium regularization strength.

(d) L2 penalty, medium regularization strength.

(e) L2 penalty, weak regularization strength.

(f) L2 penalty, weak regularization strength.

T ), ErrD(h∗

ErrDS (h∗
{

Figure 6: Results of applying L2 penalty with different strength when constructing h∗
S.
:=
and LB := lower bound speciﬁed in Theorem 4.6. The

The left column consisting of panels (a),
max
right column consisting of panels (b), (d), and (f) compares Diff := ErrD(h∗
−
T ) and UB := upper bound speciﬁed in Theorem 4.2. For each time step
ErrD(h∗
K = k, we compute and deploy the source optimal classiﬁer h∗
S and update the credit
score for each individual according to the received decision as the new reality for time step
K = k + 1.

(c), and (e) compares Max

T )(h∗
T )
}

S )(h∗
S)

T )(h∗

D CHALLENGES IN MINIMIZING INDUCED RISK

D.1 COMPUTATIONAL CHALLENGES

The literature of domain adaptation has provided us solutions to minimize the risk on the target
distribution via a nicely developed set of results Sugiyama et al. (2008; 2007); Shimodaira (2000).
This allows us to extend the solutions to minimize the induced risk too. Nonetheless we will highlight
additional computational challenges.

We focus on the covariate shift setting. The scenario for target shift is similar. For covariate shift,
recall that earlier we derived the following fact:

ED(h)[(cid:96)(h; X, Y )] = ED[ωx(h)

(cid:96)(h; x, y)]

·

This formula informs us that a promising solution that uses ωx(h) to perform reweighted ERM. Of
course, the primary challenge that stands in the way is how do we know ωx(h). There are different
(h) Zhang et al.
methods proposed in the literature to estimate ωx(h) when one has access to
(2013b); Long et al. (2016); Gong et al. (2016). How any of the speciﬁc techniques work in our
induced domain adaptation setting will be left for a more thorough future study. In this section,
we focus on explaining the computational challenges even when such knowledge of ωx(h) can be
obtained for each model h being considered during training.

D

28

012345K10−310−210−1ValueMaxLB012345K10−210−1ValueDiﬀUB012345K10−310−210−1ValueMaxLB012345K10−310−210−1ValueDiﬀUB012345K10−310−210−1ValueMaxLB012345K10−210−1ValueDiﬀUBUnder review as a conference paper at ICLR 2023

Though ωx(h), (cid:96)(h; x, y) might both be convex with respect to (the output of) the classiﬁer h, their
product is not necessarily convex. Consider the following example:
Example 1 (ωx(h)
x
model). Notice that (cid:96) is convex in h. Let

= (0, 1]. Let the true label of each
X
y)2, and let h(x) = x (simple linear
be the uniform distribution, whose density function is

(cid:96)(h; x, y) is generally non-convex). Let
(cid:1). Let (cid:96)(h; x, y) = 1

be y(x) = 1 (cid:0)x

2 (h(x)

∈ X

−

≥

1
2

·

fD =

(cid:26)1, 0 < x

1
≤
0, otherwise

. Notice that if the training data is drawn from

, then h is the linear classiﬁer

D

that minimizes the expected loss. Suppose that, since h rewards large values of x, it induces decision
subjects to shift towards higher feature values. In particular, let

(h) have density function

fD(h) =

(cid:26)2x,
0,

0 < x
1
≤
otherwise

D

Then for all x

∈ X

, ωx(h) =

fD(h)(x)
fD(x) = 2x. Notice that ωx(h) = 2x is convex in h(x) = x. Then

D

ωx(h)

·

(cid:96)(h; x, y) = 2x

1
2

·

(h(x)

y)2
−
(cid:26)x3,
x(x

y)2 =

0 < x < 1
2
1
x
1
2 ≤

≤

1)2,

−

which is clearly non-convex.

= x(x

−

Nonetheless, we provide sufﬁcient conditions under which ωx(h)
Proposition D.1. Suppose ωx(h) and (cid:96)(h; x, y) are both convex in h, and ωx(h) and (cid:96)(h; x, y)
satisfy
(cid:96)(h; x, y) is
convex.

h, h(cid:48), x, y: (ωx(h)
∀

(cid:96)(h; x, y) is in fact convex:

0. Then ωx(h)

(cid:96)(h(cid:48); x, y))

((cid:96)(h; x, y)

ωx(h(cid:48)))

−

≥

−

·

·

·

Proof. Let us use the shorthand ω(h) := ωx(h) and (cid:96)(h) := (cid:96)(h; x, y). To show that ω(h)
convex, it sufﬁces to show that for any α

(cid:96)(h) is

·

[0, 1] and any two hypotheses h, h(cid:48) we have
ω(h(cid:48))

(cid:96)(h) + (1

ω(h)

h(cid:48))

α)

α)

α

·

≤

·

·

−

·

∈

−

(cid:96)(h(cid:48))

·

ω(α

·

h + (1

α)

·

−

h(cid:48))

By the convexity of ω,

ω(α

and by the convexity of (cid:96),

(cid:96)(α

(cid:96)(α

·

h + (1

h + (1

α)

−

h + (1

α)

−

·

·

·

Therefore it sufﬁces to show that

h(cid:48))

α

·

≤

ω(h) + (1

α)

−

h(cid:48))

α

·

≤

(cid:96)(h) + (1

α)

−

ω(h(cid:48))

(cid:96)(h(cid:48))

·

·

·

·

[α

⇔

⇔

1)

ω(h) + (1
·
α(α
α(α
−
[ω(h)

1)

−

·

(cid:96)(h) + (1

·

ω(h(cid:48))]

α)
−
ω(h)(cid:96)(h)
[ω(h)

[α
·
α(α
−
ω(h(cid:48))]

·

1)
−
[(cid:96)(h)

·

α

−

α)

(cid:96)(h(cid:48))]
·
[ω(h)(cid:96)(h(cid:48)) + ω(h(cid:48))(cid:96)(h)] + α(α
(cid:96)(h(cid:48))]

ω(h)

−

0

·

·

(cid:96)(h) + (1

1)

·

−

α)
−
·
ω(h(cid:48))(cid:96)(h(cid:48))

ω(h(cid:48))
0

≤

(cid:96)(h(cid:48))

·

0

≤

·
ω(h(cid:48))]

−
[(cid:96)(h)

·
(cid:96)(h(cid:48))]

−
0

≤

⇔
By the assumed condition, the left-hand side is indeed non-negative, which proves the claim.

≥

−

−

·

This condition is intuitive when each x belongs to a rational agent who responds to a classiﬁer h to
maximize her chance of being classiﬁed as +1: For y = +1, the higher loss point corresponds to the
ones that are close to decision boundary, therefore, more
1 negative label points might shift to it,
−
resulting to a larger ωx(h). For y =
1, the higher loss point corresponds to the ones that are likely
mis-classiﬁed as +1, which “attracts” instances to deviate to.

−

D.2 CHALLENGES DUE TO THE LACK OF ACCESS TO DATA

We discuss the challenges in performing induced domain adaptation. In the standard domain adap-
tation settings, one often assumes the access to a sample set of X, which already poses challenges

29

Under review as a conference paper at ICLR 2023

when there is no access to label Y after the adaptation. Nonetheless, the literature has observed a
fruitful development of solutions Sugiyama et al. (2008); Zhang et al. (2013b); Gong et al. (2016).

One might think the above idea can be applied to our IDA setting rather straightforwardly by
(h), the induced distribution under each model h during the
assuming observing samples from
training. However, we often do not know precisely how the distribution would shift under a model
h until we deploy it. This is particularly true when the distribution shifts are caused by human
responding to a model. Therefore, the ability to “predict” accurately how samples “react” to h plays
a very important role Ustun et al. (2019). Indeed, the strategic classiﬁcation literature enables this
capability by assuming full rational human agents. For a more general setting, building robust domain
adaptation tools that are resistant to the above “prediction error” is also going to be a crucial criterion.

D

E DISCUSSIONS ON PERFORMING DIRECT INDUCED RISK MINIMIZATION

In this section, we provide discussions on how to directly perform induced risk minimization for our
induced domain adaptation setting. We ﬁrst provide a gradient descent based method for a particular
label shift setting where the underlying dynamic is replicator dynamic described in Section 5.3. Then
we propose a solution for a more general induced domain adaptation setting where we do not make
any particular assumptions on the undelying distribution shift model.

E.1 GRADIENT DESCENT BASED METHOD

Here we provide a toy example of performing direct induced risk minimization under the assumption
of label shift with underlying dynamics as the replicator dynamics described in Section 5.3.

∈

R and a binary true qualiﬁcation y

Setting Consider a simple setting in which each decision subject is associated with a 1-dimensional
1, +1
continuous feature x
. We assume label shift
}
setting, and the underlying population dynamic evolves the replicator dynamic setting described in
Section 5.3. We consider a simple threshold classiﬁer, where ˆY = h(x) = 1[X
θ], meaning that
the classiﬁer is completely characterized by the threshold parameter θ. Below we will use ˆY and
h(X) interchangeably to represent the classiﬁcation outcome. Recall that the replicator dynamics is
speciﬁed as follows:

∈ {−

≥

PD(h)(Y = y)
PDS (Y = y)

=

Fitness(Y = y)
EDS [Fitness(Y )]

(22)

PDS (Y = y)).
where EDS [Fitness(Y )] = Fitness(Y = y)PDS (Y = y) + Fitness(Y =
Fitness(Y = y) is the ﬁtness of strategy Y = y, which is further deﬁned in terms of the expected
utility Uy,ˆy of each qualiﬁcation-classiﬁcation outcome pair (y, ˆy):

y)(1

−

−

Fitness(Y = y) :=

(cid:88)

ˆy

P[ ˆY = ˆy
Y = y]
|

·

Uy,ˆy

the utility (or
each qualiﬁcation-classiﬁcation outcome
Y = y) is sampled according to a Gaussian distribution, and will be
|

where Uy,ˆy
is
combination.P(X
unchanged since we consider a label shift setting.
We initialize the distributions we specify the initial qualiﬁcation rate PDS (Y = +1). To test different
settings, we vary the speciﬁcation of the utility matrix Uy,ˆy and generate different dynamics.

reward)

for

Formulate the induced risk as a function of h To minimize the induced risk, we ﬁrst formulate
the induced risk as a function of the classiﬁer h’s parameter θ taking into account of the underlying
dynamic, and then perform gradient descent to solve for locally optimal classiﬁer h∗
T .

Recall from Section 5, under label shift, we can rewrite the induced risk as the following form:

·

EDS [(cid:96)(h; X, Y )

ED(h)[(cid:96)(h; X, Y )] =p(h)
where p(h) = PD(h)(Y = +1).
Since EDS [(cid:96)(h; X, Y )
|
S, it sufﬁces to show that the accuracy on
S.

D
a function of θ and

D

Y = +1] and EDS [(cid:96)(h; X, Y )

D

Y = +1] + (1
|

−

p(h))

·

EDS [(cid:96)(h; X, Y )

Y =
|

−

1]

1] are already functions of both h and
(h), p(h) = PD(h)(Y = +1), can also be expressed as

Y =
|

−

30

Under review as a conference paper at ICLR 2023

To see this, recall that for a threshold classiﬁer ˆY = 1[X > θ], it means that the prediction accuracy
can be written as a function of the threshold θ and target distribution

(h):

D

PD(h)(Y = +1)

= PD(h)( ˆY = +1, Y = +1) + PD(h)( ˆY =
= PD(h)(X
(cid:90) ∞

θ, Y = +1) + PD(h)(X

≤
PD(h)(Y = +1) P(X = x
Y = 1)
|
(cid:125)
(cid:123)(cid:122)
unchanged because of label shift

=

θ

≥

(cid:124)

−
θ, Y =

dx

1, Y =

1)

−
1)

−

(cid:90) θ

+

−∞

PD(h)(Y =

−

1) P(X = x
Y =
|
(cid:123)(cid:122)
unchanged because of label shift

1)
(cid:125)

−

(cid:124)

dx

(23)

where P(X
Y = y) remains unchanged over time, and PD(h)(Y = y) evolves over time according
|
to Equation (22), namely

PD(h)(Y = y)

=PDS (Y = y)

=PDS (Y = y)

×

×

Fitnessg(Y = y)
EDS [Fitnessg(Y )]
(cid:80)
ˆy

(cid:80)

y((cid:80)

ˆy

PDS [ ˆY = ˆy
|

PDS [ ˆY = ˆy
|

Y = y, G = g]

Uˆy,y

Y = y, G = g]

Uˆy,y)PDS [Y = y]

·

·

(24)

Notice that ˆY is only a function of θ, and Uy,ˆy are ﬁxed quantities, the above derivation indicates that
we can express PD(h)(Y = y) as a function of θ and
S. Plugging it back to Equation (23), we can
see that the accuracy can also be expressed as a function of the classiﬁer’s parameter θ, indicating
that the induced risk can be expressed as a function of θ. Thus we can use gradient descent using
automatic differentiation w.r.t θ to ﬁnd a optimal classiﬁer h∗

T that minimize the induced risk.

D

Figure 7: Experimental results of directly optimizing for the induced risk under the assumption of
replicator dynamic. The X-axis denotes the prediction accuracy of ErrD(h∗
S is the
source optimal classiﬁer under each settings. The Y-axis is the percent of performance improvement
using the classiﬁer that optimize for h∗
T = arg min ErrD(h)(h), which the decision maker considers
the underlying response dynamics (according to replicator dynamics in Equation (22)) of the decision
subjects. Different color represents different utility function, which is reﬂected by the speciﬁcations
of values in Uy,ˆy; within each color, different dots represent different initial qualiﬁcation rate.

S), where h∗

S )(h∗

Experimental Results Figure 7 shows the experimental results for this toy example. We can see
that for each setting, compared to the baseline classiﬁer h∗
S, the proposed gradient based optimization

31

Under review as a conference paper at ICLR 2023

procedure returns us a classiﬁer that achieves a better prediction accuracy (thus lower induced risk)
compared to the accuracy of the source optimal classiﬁer.

E.2 GENERAL SETTING: INDUCED RISK MINIMIZATION WITH BANDIT FEEDBACK

D

In general, ﬁnding the optimal classiﬁer that achieves the optimal induced risk h∗
T is a hard problem
due to the interactive nature of the problem (see, e.g. the literature of performative prediction Perdomo
et al. (2020) for more detailed discussions). Without making any assumptions on the mapping between
(h), one can only potentially rely on the bandit feedbacks from the decision subjects to
h and
estimate the inﬂuence of h on
(h): when the induced risk is a convex function of the classiﬁer
h’s parameter θ, one possible approach is to use the standard techniques from bandit optimization
(Flaxman et al., 2004) to iteratively ﬁnd induced optimal classiﬁer h∗
T . The basic idea is: at each step
t = 1,
(ht)
and their losses, and use them to construct an approximate gradient for the induced risk as a function
of the model parameter θt. When the induced risk is a convex function in the model parameter θ, the
above approach guarantees to converge to h∗
T , and have sublinear regret in the total number of steps
T .
The detailed description of the algorithm for ﬁnding h∗

, T , the decision maker deploy a classiﬁer ht, then observe data points sampled from

· · ·

D

D

T is as follows:

Algorithm 1: One-point bandit gradient descent for performative prediction
Result: return θT after T rounds
θ1
foreach time step t

0

∼

←

Unif(S)

1, . . . , T do

←
Sample a unit vector ut
θ+
θt + δut
t ←
Observe data points z1, . . . , znt ∼ D
(cid:80)nt
(cid:101)IR(θ+
i=1 (cid:96)(zi; θ+
1
t )
t )
nt
δ (cid:101)IR(θ+
ut
˜gt(θt)
t )
·
θt+1
Π(1−δ)Θ(θt
(1

←
δ)Θ :=

−
δ)θ

η˜gt(θt))
θ
Θ
}

←
d

←

(1

−

−

∈

{

|

end

(θ+
t )

(cid:46) ˜gt(θt) is an approximation of

θ (cid:98)IR(θt)
(cid:46) Take gradient step; project onto

∇

F REGULARIZED TRAINING

In this section, we discuss the possibility that indeed minimizing regularized risk will lead to a tighter
upper bound. Consider the target shift setting. Recall that p(h) := PD(h)(Y = +1) and we have for
any proper loss function (cid:96):
ED(h)[(cid:96)(h; X, Y )] = p(h)

Y = +1] + (1

p(h))

Y =

1]

EDS [(cid:96)(h; X, Y )
|

·

−

EDS [(cid:96)(h; X, Y )
|

·

−

Suppose p < p(h∗
a smaller upper bound.

T ), now we claim that minimizing the following regularized/penalized risk leads to

EDS [(cid:96)(h; X, Y )] + α

h(X) + 1
2
uniform is a distribution with uniform prior for Y .

EDuniform||

·

||

where in above

D

We impose the following assumption:

• The number of predicted +1 for examples with Y = +1 and for examples with Y =

are monotonic with respect to α.

1

−

Consider the easier setting with (cid:96) = 0-1 loss. Then

EDuniform||

h(X)

||

= 0.5
= 0.5

·

·

(PX|Y =+1[h(X) = +1] + PX|Y =−1[h(X) = +1])
(EX|Y =+1[(cid:96)(h(X), +1)]
1])

EX|Y =−1[(cid:96)(h(X),

−

0.5

−

−

32

Under review as a conference paper at ICLR 2023

The above regularized risk minimization problem is equivalent to

(p + 0.5

α)

EX|Y =+1[(cid:96)(h(X), +1)] + (p

0.5

α)

·

·

−

EX|Y =−1[(cid:96)(h(X),

1]

−

·

·
Recall the upper bound in Theorem 5.1:
T )(h∗
S )(h∗
T )
S)

ErrD(h∗

ErrD(h∗

−

p(h∗
T )
|
(cid:125)

−
(cid:123)(cid:122)
Term 1

p(h∗
S)
≤ |
(cid:124)
+(h∗

S),

+ (1 + p)

·

(dTV(
(cid:124)

D

+(h∗

D

T )) + dTV(
(cid:123)(cid:122)
Term 2

−(h∗

S),

D

D

−(h∗

.

T ))
(cid:125)

With a properly speciﬁed α > 0, this leads to a distribution with a smaller gap of
,
T )
|
where ˜hS denotes the optimal classiﬁer of the penalized risk minimization - this leads to a smaller
Term 1 in the bound of Theorem 5.1. Furthermore, the induced risk minimization problem will
correspond to an α s.t. α∗ = p(h∗
, and the original h∗
S corresponds to a distribution of α = 0.
Using the monotonicity assumption, we will establish that the second term in Theorem 5.1 will also
smaller when we tune a proper α.

T )−p
0.5

−

p(h∗

p(˜hS)
|

G DISCUSSION ON THE TIGHTNESS OF OUR THEORETICAL BOUNDS

General Bounds in Section 3 For the general bounds reported in Section 3, it is not trivial to fully
quantify the tightness without further quantifying the speciﬁc quantities of the terms, e.g. the H
divergence of the source and the induced distribution, and the average error a classiﬁer have to incur
for both distribution. This part of our results adapted from the classical literature in learning from
multiple domains Ben-David et al. (2010). The tightness of using
-divergence and other terms seem
to be partially validated therein.

H

Bounds in Section 4 and Section 5 For more speciﬁc bounds provided in Section 4 (for covariate
shift) and Section 5 (target shift), however, it is relatively easier to argue about the tightness: the
proofs there are more transparent and are easier to back out the conditions where the inequalities are
relaxed. For example, in Theorem 5.1, the inequalities of our bound are introduced primarily in the
following two places: 1) one is using the optimiality of h∗
S on the source distribution. 2) the other is
bounding the statistical difference in h∗
T ’s predictions on the positive and negative examples.
Both are saying that if the differences in the two classiﬁers’ predictions are bounded in a range, then
the result in Theorem 5.1 is relatively tight.

S and h∗

33

