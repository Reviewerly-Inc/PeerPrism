Published as a conference paper at ICLR 2021

PROVABLY ROBUST CLASSIFICATION OF
ADVERSARIAL EXAMPLES WITH DETECTION

Fatemeh Sheikholeslami
Bosch Center for Artiﬁcial Intelligence
Pittsburgh, PA
fatemeh.sheikholeslami@us.bosch.com

Ali Lotﬁ Rezaabad ∗
The University of Texas at Austin
Austin, TX
alotfi@utexas.edu

J. Zico Kolter
Bosch Center for Artiﬁcial Intelligence
Carnegie Mellon University
Pittsburgh, PA
zkolter@cs.cmu.edu

ABSTRACT

Adversarial attacks against deep networks can be defended against either by build-
ing robust classiﬁers or, by creating classiﬁers that can detect the presence of ad-
versarial perturbations. Although it may intuitively seem easier to simply detect
attacks rather than build a robust classiﬁer, this has not bourne out in practice
even empirically, as most detection methods have subsequently been broken by
adaptive attacks, thus necessitating veriﬁable performance for detection mecha-
nisms. In this paper, we propose a new method for jointly training a provably
robust classiﬁer and detector. Speciﬁcally, we show that by introducing an addi-
tional “abstain/detection” into a classiﬁer, we can modify existing certiﬁed defense
mechanisms to allow the classiﬁer to either robustly classify or detect adversarial
attacks. We extend the common interval bound propagation (IBP) method for cer-
tiﬁed robustness under (cid:96)∞ perturbations to account for our new robust objective,
and show that the method outperforms traditional IBP used in isolation, especially
for large perturbation sizes. Speciﬁcally, tests on MNIST and CIFAR-10 datasets
exhibit promising results, for example with provable robust error less than 63.63%
and 67.92%, for 55.6% and 66.37% natural error, for (cid:15) = 8/255 and 16/255 on
the CIFAR-10 dataset, respectively.

1

INTRODUCTION

Despite popularity and success of deep neural networks in many applications, their performance
declines sharply in adversarial settings. Small adversarial perturbations are shown to greatly dete-
riorate the performance of neural network classiﬁers, which creates a growing concern for utilizing
them in safety critical application where robust performance is key. In adversarial training, different
methods with varying levels of computational complexity aim at robustifying the network by ﬁnding
such adversarial examples at each training steps and adding them to the training dataset. While such
methods exhibit empirical robustness, they lack veriﬁable guarantees as it is not provable that a more
rigorous adversary, e.g., one that does brute-force enumeration to compute adversarial perturbations,
will not be able to cause the classiﬁer to misclassify.

It is thus desirable to provably verify the performance of robust classiﬁers without restricting the ad-
versarial perturbations by inexact solvers, while restraining perturbations to a class of admissible set,
e.g., within an (cid:96)∞ norm-bounded ball. Progress has been made by ‘complete methods’ that use Sat-
isﬁability Modulo Theory (SMT) or Mixed-Integer Programming (MIP) to provide exact robustness
bounds, however, such approaches are expensive, and difﬁcult to scale to large networks as exhaus-
tive enumeration in the worst case is required (Tjeng et al., 2017; Ehlers, 2017; Xiao et al., 2018).

∗Work was done when the author was an intern at Bosch Center for Artiﬁcial Intelligence, Pittsburgh, PA.

1

Published as a conference paper at ICLR 2021

‘Incomplete methods’ on the other hand, proceed by computing a differential upper bound on the
worst-case adversarial loss, and similarly for the veriﬁcation violations, with lower computational
complexity and improved scalability. Such upper bounds, if easy to compute, can be utilized during
the training, and yield provably robust networks with tight bounds. In particular, bound propaga-
tion via various methods such as differentiable geometric abstractions (Mirman et al., 2018), convex
polytope relaxation (Wong & Kolter, 2018), and more recently in (Salman et al., 2019; Balunovic &
Vechev, 2020; Gowal et al., 2018; Zhang et al., 2020), together with other techniques such as semi-
deﬁnite relaxation, (Fazlyab et al., 2019; Raghunathan et al., 2018), and dual solutions via additional
veriﬁer networks (Dvijotham et al., 2018) fall within this category. In particular, recent successful
use of Interval Bound Propagation (IBP) as a simple layer-by-layer bound propagation mechanism
was shown to be very effective in Gowal et al. (2018), which despite its light computational com-
plexity exhibits SOTA robustness veriﬁcation. Additionally, combining IBP in a forward bounding
pass with linear relaxation based backward bounding pass (CROWN) Zhang et al. (2020) leads to
improved robustness, although it can be up to 3-10 times slower.

Alternative to robust classiﬁcation, detection of adversarial examples can also provide robust-
ness against adversarial attacks, where suspicious inputs will be ﬂagged and the classiﬁer “re-
jects/abstains” from assigning a label. There has been some work on detection of out-of-distribution
examples Bitterwolf et al. (2020), however the situation in the literature on the detection of adver-
sarial examples is quite different from above. Most techniques that attempt to detect adversarial
examples, either by training explicit classiﬁers to do so or by simply formulating “hand-tuned” de-
tectors, still largely look to identify and exploit statistical properties of adversarial examples that
appear in practice Smith & Gal (2018); Roth et al. (2019). However, to provide a fair evalua-
tion, a defense must be evaluated under attackers that attempt to fool both the classiﬁer and the
detector, while addressing particular characteristics of a given defense, e.g., gradient obfuscation,
non-differentability, randomization, and simplifying the attacker’s objective for increased efﬁciency.
A non-exhaustive list of recent detection methods entails randomization and sparsity-based defenses
(Xiao et al., 2019; Roth et al., 2019; Pang et al., 2019b), conﬁdence and uncertainty-based detection
(Smith & Gal, 2018; Stutz et al., 2020; Sheikholeslami et al., 2020), transformation-based defenses
(Bafna et al., 2018; Yang et al., 2019), ensemble methods (Verma & Swami, 2019; Pang et al.,
2019a), generative adversarial training Yin et al. (2020), and many more. Unfortunately, existing
defenses have largely proven to have poor performance against adaptive attacks (Athalye et al., 2018;
Tramer et al., 2020), necessitating provable guarantees on detectors as well. Recently Laidlaw &
Feizi (2019) have proposed joint training of classiﬁer and detector, however it also does not provided
any provable guarantees.

Our contribution. In this work, we propose a new method for jointly training a provably robust
classiﬁer and detector. Speciﬁcally, by introducing an additional “abstain/detection” into a classi-
ﬁer, we show that the existing certiﬁed defense mechanisms can be modiﬁed, and by building on
the detection capability of the network, classiﬁer can effectively choose to either robustly classify or
detect adversarial attacks. We extend the light-weight Interval Bound Propagation (IBP) method to
account for our new robust objective, enabling veriﬁcation of the network for provable performance
guarantees. Our proposed robust training objective is also effectively upper bounded, enabling its
incorporation into the training procedure leading to tight provably robust performance. While tight-
ening of the bound propagation may be additionally possible for tighter veriﬁcation, to the best of
our knowledge, our approach is the ﬁrst method to extend certiﬁcation techniques by considering
detection while providing provable veriﬁcation. By stabilizing the training, as also used in simi-
lar IBP-based methods, experiments on MNIST and CIFAR-10 empirically show that the proposed
method can successfully leverage its detection capability, and improves traditional IBP used in iso-
lation, especially for large perturbation sizes.

2 BACKGROUND AND RELATED WORK

Let us consider an L-layer feed-forward neural network, trained for a K-class classiﬁcation task.
Given input x, it will pass through a sequential model, with hl denoting the mapping at layer l,
recursively parameterized by

zl = hl(zl−1) = σl(W(cid:62)

(1)
where σl(.) is a monotonic activation function, z0 denotes the input, and zL ∈ RK is the pre-
activation unnormalized K-dimensional output vector (nL = K and σL(.) as identity operator),

l zl−1 + bl), l = 1, · · · , L Wl ∈ Rnl−1×nl , bl ∈ Rnl

2

Published as a conference paper at ICLR 2021

referred to as the logits. Robust classiﬁers can be obtained by minimizing the worst-case (adversar-
ial) classiﬁcation loss, formally trained by the following min-max optimization Madry et al. (2017)

minimize
θ

E
(x,y)∼D

(cid:20)

max
δ∈∆(cid:15)

(cid:21)

(cid:96)(fθ(x + δ), y)

.

(2)

where θ denotes network parameters, vector fθ(x) = zL is the logit output for input x, (cid:96)(.) is
the misclassﬁcation loss, e.g., (cid:96)xent(.) deﬁned as the cross-entropy loss, and ∆(cid:15) denotes the set
of permissible perturbations, e.g., for (cid:96)∞-norm ball of radius (cid:15) giving ∆(cid:15) := {δ | (cid:107)δ(cid:107)∞ ≤ (cid:15)}.
Although augmenting the training set with adversarial inputs, obtained by approximately solving
the inner maximization in Eq. 2, empirically leads to improved adversarial robustness Madry et al.
(2017); Shafahi et al. (2019); Wong et al. (2019); Zhang et al. (2019), inexact solution of the inner
maximization prevents such methods from providing provable guarantees. In critical applications
however, provable veriﬁcation of classiﬁcation accuracy against a given threat model is crucial.

2.1

PERFORMANCE VERIFICATION AND NETWORK RELAXATION

Given input (x, y), a classiﬁcation network is considered veriﬁably robust if all of its perturbed
variations, that is x + δ for ∀δ ∈ ∆(cid:15), are correctly classiﬁed as class y. Such veriﬁcation can be
effectively obtained by
p∗
i = min
zL∈ZL

c(cid:62)
y,izL where , ZL := {zL|zl = hl(zl−1), l = 1, ..., L, z0 = x + δ, ∀δ ∈ ∆(cid:15)}

where cy,i = ey − ei for i = 1, 2, .., K, i (cid:54)= y, and ei is the standard ith canonical basis vector.
If p∗
i > 0 ∀i (cid:54)= y, then the classiﬁer is veriﬁably robust at point (x, y) as this guarantees that
zi ≤ zy ∀i (cid:54)= y for all admissible perturbations δ ∈ ∆(cid:15).
The feasible set ZL is generally nonconvex, rendering obtaining p∗
i intractable. Any convex relax-
ation of ZL however, will provide a lower bound on p∗
i , and can be alternatively used for veriﬁcation.
As outlined in Section 1, various relaxation techniques have been proposed in the literature. Specif-
ically, IBP in (Mirman et al., 2018; Gowal et al., 2018) proceeds by bounding the activation zl of
each layer by propagating an element-wise bounding box using interval arithmetic for networks with
monotonic activation functions. Despite its simplicity and relatively small computational complex-
ity (computational requirements for bound propagation for a given input using IBP is equal to two
forward passes of the input), it can provide tight bounds once the network is trained accordingly.

Speciﬁcally, starting from the input layer z0, it can be bounded for the perturbation class δ ∈ ∆(cid:15) as
z0 = x − (cid:15)1 and z0 = x + (cid:15)1, and zl for the following layers can be bounded as
zl−1 − zl−1
2

zl−1 + zl−1
2

zl−1 + zl−1
2

zl−1 − zl−1
2

zl = σl(W(cid:62)

zl = σl(W(cid:62)
l

− |W(cid:62)
l |

+ |W(cid:62)
l |

),

),

l

(3)

where | · | is the element-wise absolute-value operator. The veriﬁcation problem over the relaxed
feasible set ˆZL := {zL | zL,i ≤ zL,i ≤ ¯zL,i}, where ZL ⊆ ˆZL is then easily solved as
c(cid:62)
y,izL ≥ min
zL∈ ˆZL

c(cid:62)
y,izL = zL,y − ¯zL,i.

p∗
i = min
zL∈ZL

(4)

2.2 ROBUST TRAINING OF VERIFIABLE NETWORKS

It has been shown that convex relaxation of ZL can also provide a tractable upper bound on the inner
maximization in Eq. 2. While this holds for various relaxation techniques, focusing on the IBP let
us deﬁne

(cid:15),θ (x, y) := [J IBP
JIBP

1

, J IBP
2

, ..., J IBP

K ] where

J IBP
i

:= min
zL∈ ˆZL

c(cid:62)
y,izL

(5)

with (θ, (cid:15)) implicitly inﬂuencing ˆZL (dropped for brevity), and upperbound the inner-max in Eq. 2

max
δ∈∆(cid:15)

(cid:96)xent(fθ(x + δ), y) ≤ (cid:96)xent(−JIBP

(cid:15),θ (x, y), y),

(cid:96)xent(z, c) := − log

(cid:16) exp(zc)

(cid:17)

(cid:80)

i exp(zi)

By using this tractable upper bound of the robust optimization, network can now be trained by

minimize
θ

(cid:88)

(x,y)∈D

(1 − κ)(cid:96)xent(−JIBP

(cid:15),θ (x, y), y) + κγ(cid:96)xent(fθ(x), y),

3

(6)

(7)

Published as a conference paper at ICLR 2021

where γ trades natural versus robust accuracy, and κ is scheduled through a ramp-down process to
stabilize the training and tightening of IBP Gowal et al. (2018) (where γ = 1 is selected therein).

3 VERIFIABLE CLASSIFICATION WITH DETECTION

In this paper, we propose a new method for jointly training a provably robust classiﬁer and detector.
Speciﬁcally, let us augment the classiﬁer by introducing an additional “abstain/detection”. This can
be readily done by extending the K-class classiﬁcation task to a (K + 1)-class classiﬁcation, with
the (K + 1)-th class dedicated to the detection task, and the maximum weighted class is ﬁnally
chosen as the classiﬁcation output. The classiﬁer is then trained such that adversarial examples, or
ideally any other example that the network would misclassify, are classiﬁed in this abstain class,
denoted by a, thus preventing incorrect classiﬁcation.

Formally, the classiﬁer can be denoted as in Eq. 1, with the only difference that the ﬁnal output is
K + 1 dimensional, i.e., zL ∈ RK+1; simply by substituting the last fully-connected weight matrix
WL of dimension nL × K with that of dimension nL × (K + 1), and similarly for bL.

3.1 VERIFICATION PROBLEM FOR CLASSIFICATION WITH ABSTAIN/DETECTION

It is desirable to provably verify performance of the joint classiﬁcation/detection. In contrast to
existing robust classiﬁers, however, on a perturbed image x + δ, the classiﬁcation/detection task is
considered successful if the input is classiﬁed either as the correct class y, or as the abstain class
a; as both cases prevent misclassiﬁcation of the adversarially perturbed input as a wrong class. On
clean natural images however, classiﬁcation/detection is considered successful only if it is classiﬁed
as the correct class y, and abstaining is considered misclassiﬁcation.

In order to certify performance in adversarial settings, it is now sufﬁcient to verify that the network
satisﬁes the following for a given input pair (x, y) and δ ∈ ∆(cid:15) and i = 1, .., K, i (cid:54)= y:
max{c(cid:62)

∀z ∈ ZL := {zL|zl = hl(zl−1) l = 1, ..., L, z0 = x + δ, ∀δ ∈ ∆(cid:15)}
(8)
where cy := ey − ei and ca := ea − ei, a denotes the “abstain” class, and the dependence of ZL on
(x, y, (cid:15), θ) is omitted for brevity. Veriﬁcation can be done effectively by seeking a counterexample

a,iz} ≥ 0

y,iz, c(cid:62)

π∗
i := min
z∈ZL

max{c(cid:62)

y,iz, c(cid:62)

a,iz}.

(9)

If π∗

i ≥ 0 ∀i (cid:54)= y , the speciﬁcation is then satisﬁed and the performance is veriﬁed.

Similar to previous veriﬁcation methods, to overcome the non-convexity of the optimization in Eq.
9, one can lower bound the problem by expanding the feasible set ZL ⊆ ˆZL , where ˆZL is convex,
as stated in Theorem 1, and proved in Appendix A.1.
Theorem 1: For any convex ˆZL s.t. ZL ⊆ ˆZL, Eq. 9 can be bounded by the convex relaxation

max
0≤η≤1

min
z∈ ˆZL

(cid:16)

η ca,i + (1 − η) cy,i

(cid:17)(cid:62)

z ≤ min
z∈ZL

max{c(cid:62)

y,iz, c(cid:62)

a,iz}.

(10)

Although Theorem 1 holds for any convex relaxation of ZL, for IBP relaxation in Gowal et al.
(2018) it can be further simpliﬁed by substituting z = W(cid:62)
L zL−1 + bL, thus not propagating the
intervals through the last layer for tighter bounding, and solved analytically as follows.

Theorem 2: The optimization in Eq. 9 can be lower-bounded by the convex optimization
y,iz, c(cid:62)
Ji(x, y) = max
0≤η≤1

min
zL−1∈ ˆZL−1
L cy,i, ω2 := W(cid:62)

(ω1 + η ω2)(cid:62)zL−1 + η ω3 + ω4 ≤ min
z∈ZL
L (ca,i − cy,i), ω4 := b(cid:62)

in which ω1 := W(cid:62)
L (ca,i − cy,i), ω3 := b(cid:62)
L cy,i and convex
set ˆZL−1 is a convex relaxation of ZL−1 on the hidden values at L − 1. Furthermore, Ji(x, y) can
be analytically obtained as outlined in Alg. 1

a,iz} (11)

max{c(cid:62)

Note that since η is the dual variable, any selection within the feasible set serves as a (looser but
valid) lower bound, while the maximization makes the bound tight; see Appendix A.2 and A.3 for
proof and a step-by-step algorithm description. Similar to other convex relaxation-based veriﬁcation
methods, in order for a networks to provide veriﬁable performance, one needs to incorporate bound
propagation in training.

4

Published as a conference paper at ICLR 2021

Algorithm 1 Solution for Ji(x, y) in Theorem 2
1: Input. Bounds on layer L − 1 : zL−1, ¯zL−1, and weight matrix WL
2: ω1 = WLcy,i and ω2 = WL(ca,i − cy,i), ω3 := b(cid:62)
3: ζ = [ζ1, ..., ζnL] := −ω1/ω2 and vector of indices s that sorts ζ , i.e., ζs1 ≤ · · · ≤ ζsnL−1
4: u1 = Πs(ω1 ◦ zL−1) , ¯u1 = Πs(ω1 ◦ ¯zL−1), u2 := Πs(ω2 ◦ zL−1) , ¯u2 := Πs(ω2 ◦ ¯zL−1)
where operators ◦ and Πs(.) denote element-wise multiplication, and permutation according to
indices s, respectively.

L (ca,i − cy,i), and ω4 := b(cid:62)

L cy,i

5: m = minζsj ≥0 j
6: for η = 0, ζsm, ζsm+1, · · · , ζsM −1 , ζsM , 1 do
7:

and M = maxζsj ≤1 j

Compute

for

j = 1, ..., nL−1

g(η) =

nL−1
(cid:88)

(cid:16)

j=1

1{ω1,j +ηω2,j ≤0} (¯u1,j + η¯u2,j) + 1{ω1,j +ηω2,j ≥0}

(cid:0)u1,j + ηu2,j

(cid:1) (cid:17)

+ η ω3 + ω4

8: return max g(η) over the computed values.

4 TRAINING A VERIFIABLE ROBUST CLASSIFICATION WITH DETECTION

In order to train a robust classiﬁer with detection, let us start by formalizing the objective of an
adversarial attacker. Naturally, an adaptive attacker’s objective is to craft perturbation δ such that
it simultaneously evades detection and causes misclassiﬁcation. Formally, this can be tackled by
seeking δ such that loss corresponding to the winner of the two classes y and a (higher logit leading
to smaller cross-entropy loss) is maximized, i.e.,

max
δ∈∆

min

(cid:110)

(cid:96)xent(fθ(x + δ), y), (cid:96)xent(fθ(x + δ), a)

(cid:111)

(12)

where (cid:96)xent(z, c) denotes the cross-entropy loss for class c = y and c = a, and I = {1, 2, ..., K, a}
denotes the class index set with K + 1 elements.

Let us now deﬁne

Labstain

robust (x, y; θ) := max
δ∈∆

min

(cid:110)

(cid:96)xent\a(fθ(x + δ), y), (cid:96)xent\y(fθ(x + δ), a)

(cid:111)

(13)

in which the inner maximization is closely related to that of the adversarial objective in Eq. 12 with
a small difference: loss terms (cid:96)xent\a and (cid:96)xent\y are deﬁned as

(cid:96)xent\a(z, y) := − log

(cid:16)

exp(zy)
(cid:80)

exp(zi)

(cid:17)

i∈I\{a}

, and (cid:96)xent\y(x, a) := − log

(cid:16)

exp(za)
(cid:80)

exp(zi)

(cid:17)

.

i∈I\{y}

This small alteration to the cost, while not changing the minimization “winner” between the true
class y and rejection class a in Eq. 12 and 13, i.e.,
(cid:26)za ≤ zy ⇒ (cid:96)xent(fθ(x + δ), y) ≤ (cid:96)xent(fθ(x + δ), a) and (cid:96)xent\a(fθ(x + δ), y) ≤ (cid:96)xent\y(fθ(x + δ), a)
zy ≤ za ⇒ (cid:96)xent(fθ(x + δ), a) ≤ (cid:96)xent(fθ(x + δ), y) and (cid:96)xent\y(fθ(x + δ), a) ≤ (cid:96)xent\a(fθ(x + δ), y)

favorably inﬂuences the training process. That is so since, for δ such that, for instance za < zy,
minimizing Labstain
robust (x, y; θ) during training reduces to minimizing (cid:96)xent(fθ(x + δ), y) which in turn
leads to further increasing zy while decreasing the logit value za; and similarly, increasing zy while
decreasing zy if zy < za. Intuitively however, the true objective of the classiﬁer augmented with
detection on adversarial examples is to increase both zy and za while reducing zj, ∀j (cid:54)= a, y; thus
preventing any gap in between the boundary of the classes a and y, which can potentially lead to
successful adaptive attacks. Hence, minimizing Eq. 12 would be in contrast with the true underlying
objective, and Eq. 13 simply prevents the raised issue.

Upon deﬁning Lnatural(x, y; θ) := (cid:96)xent(fθ(x), y) and Lrobust(x, y; θ) := maxδ∈∆ (cid:96)xent(fθ(x + δ), y),
we then deﬁne the overall training loss as

L = Lrobust(x, y; θ) + λ1Labstain

robust (x, y; θ) + λ2Lnatural(x, y; θ),

(14)

5

Published as a conference paper at ICLR 2021

where Lnatural(x, y; θ) captures the misclassiﬁcation loss of the natural (clean) examples, and
Lrobust(x, y; θ) denotes that of adversarial examples without considering the rejection class, i.e., sim-
ilar to that of Gowal et al. (2018), and parameters (λ1, λ2) trade-off clean and adversarial accuracy.
To train a robust classiﬁer, we proceed by minimizing the overall loss Eq. 14, by ﬁrst upperbounding
Lrobust(x, y; θ) and Labstain

robust (x, y; θ).

4.1 UPPERBOUNDING THE TRAINING LOSS

Using Theorem 2, and restricting 0 < η ≤ η ≤ ¯η < 1, let us now deﬁne J

η,¯η
i

(x, y), where trivially

η,¯η
J
i

(x, y) := max

0≤η≤η≤¯η≤1

(ω1 + ηω2)(cid:62)ˆzL−1 + η ω3 + ω4 ≤ Ji(x, y)

(15)

and can also be solved analytically similar to Theorem 2. By generalizing the ﬁndings in Wong &
Kolter (2018); Mirman et al. (2018), we can upper bound the robust optimization problem using our
dual problem in Eq. 15, according to the following Theorem, which we prove in Appendix A.4.

Theorem 3: For any data point (x, y), and (cid:15) > 0, and for any 0 ≤ η ≤ ¯η ≤ 1, the adversarial loss
Labstain

robust (x, y; θ) in Eq. 13 can be upper bounded by

robust (x, y; θ) ≤ ¯Labstain
Labstain

robust (x, y; θ) := (cid:96)xent\a(−J(cid:15),θ(x, y), y) = (cid:96)xent\y(−J(cid:15),θ(x, y), a)
η,¯η
where J(cid:15),θ(x, y) is a (K + 1)-dimensional vector, valued at index i as [J(cid:15),θ(x, y)]i = J
i

(16)

(x, y).

η,¯η
Note that maximization over η for obtaining J
i
maximization) or by following Alg. 1 and substituting m = minζsν ≥η ν
Remark 1. Setting η = ¯η = 0 forces η = 0 which reduces J
J IBP
i

(x, y)|η=¯η=0, also bounding loss term Lnatural(x, y; θ) as

(x, y) can be done either by bisection (concave
, and M = maxζsν ≤ ¯η ν
(x, y) in Eq. 15 to that in Eq. 5, .i.e,

η,¯η
(x, y) = J
i

η,¯η
i

Lrobust(x, y; θ) ≤ ¯Lrobust(x, y; θ) := (cid:96)xent(−JIBP

(cid:15),θ (x, y), y).

(17)

Remark 2. While setting η = 0 and ¯η = 1 gives tighter bounds, (and is thus used for the veriﬁcation
counterpart in Theorem 2), strictly setting 0 < η ≤ ¯η < 1 empirically yields better generalization of
the network. This can be intuitively understood by rewriting ω1 + ηω2 = W(cid:62)
L (ηca,i + (1 − η)cy,i)
which is a convex combination of the veriﬁcation constraints for the correct and the abstain class.
Thus η (cid:54)= 0 (cid:54)= 1 will lead to minimizing a combination of both terms, preventing gaps in between
the two classes. Also, higher values of η increase the inﬂuence of the term corresponding to the
abstain case, and vice versa, whose tuning can promote abstaining by considering how desirable
such outcome is (or is not).

Utilizing upperbounds in Eq. 16 and Eq. 17, we can proceed to training the network by minimizing
the tractable upperbound on the overall loss

min
θ

L ≤ min

θ

(cid:96)xent(−JIBP

(cid:15),θ (x, y), y) + λ1(cid:96)xent\y(−J(cid:15),θ(x, y), a) + λ2(cid:96)natural(fθ(x), y)

(18)

Note that setting λ1 = 0 and γ = λ2 - and incorporation of a ramp-down process by parameter κ as
detailed in Section 5 - reduces the training in Eq. 18 to that of Gowal et al. (2018) without detection.

Complexity. Since given IBP bounds on zL−1, the solution to Eq. 16 is analytically available (that
is after sorting whose complexity is negligible in comparison with forward pass), computing Eq. 18
imposes the same computational complexity as in IBP, which is twice the normal training procedure,
as it requires propagating the upper and lower bounds via forward pass.

5 EXPERIMENTS

Empirical performance of the proposed robust classiﬁcation with detection on MNIST-10 and
CIFAR-10 datasets is reported in this section, and is compared with the state-of-the-art alternatives.
The training procedure is stabilized as detailed next 1.

1Code is available at https://github.com/boschresearch/robust_classification_

with_detection

6

Published as a conference paper at ICLR 2021

5.1 STABILIZING THE TRAINING PROCEDURE

We incorporate the following mechanisms to stabilize the training procedure in our tests, where the
ﬁrst two have been previously used in (Gowal et al., 2018) and (Zhang et al., 2020) as well.

Ramp down of κ: To stabilize the trade-off between nominal and veriﬁed accuracy, let us introduce
parameter κ in the overall loss by trading the natural and robust loss as

L = (1 − κ)

(cid:16) ¯Lrobust(x, y; θ) + λ1 ¯Labstain
(cid:124)

(cid:17)
robust (x, y; θ)
(cid:125)

(cid:123)(cid:122)
Robust loss

+ κ λ2Lnatural(x, y; θ)
(cid:125)

(cid:123)(cid:122)
Natural loss

(cid:124)

(19)

Setting κ = 0.5 renders the optimization identical to that in Eq. 18. During the training however, we
incorporate a ramp down procedure where κ starts at value κstart = 1, thus training the model to ﬁt
the natural data, and slowly decreasing it to value κend = 0.5, similar to that in Gowal et al. (2018).

Ramp up of (cid:15): It is very important during the training process to start at (cid:15) = 0 and gradually
increase it to (cid:15)train, while also setting (cid:15)train larger than (cid:15)test can improve generalization.

Ramp down of η and ¯η: Setting 0 < η and ¯η < 1 helps with better generalization. Furthermore,
setting large η and ¯η promotes the abstain class in loss term ¯Labstain
robust by increasing the weight of ω2
in Eq. 15. Thus, we can further stabilize the training process through a ramp down procedure where
these parameters start at η = η
and ¯η = ¯ηend,
with η

and ¯η = ¯ηstart, and are gradually reduced to η = η

start
and ¯ηend < ¯ηstart.

< η

end

end

start

Furthermore, although the term ¯Lrobust(x, y; θ) could in theory be excluded from the training process,
as the term Lnatural(x, y; θ) prevents the degenerate solution of always classifying all images in the
abstain class, it’s inclusion empirically helps the stability of the training process.

5.2 EMPIRICAL RESULTS ON MNIST AND CIFAR10

The classiﬁcation networks are identical to the large network in Gowal et al. (2018), also detailed
in Table 2, trained by minimizing the loss in Eq. 18 with the above stabilizing schemes. Selec-
tion of parameters for each datasets is detailed in Appendix B. Since most recent detector networks
have shown very low performance against adaptive attacks, and lack provable performance Tramer
et al. (2020), we only compare the performance with other provable robust classiﬁcation methods,
while focusing on the different decomposition in the reported natural and robust accuracy among
these two. As numbers in Table 1 suggest, the proposed detection/classiﬁcation network shows
improved robustness against other methods, including IBP in isolation (without the detection capa-
bility), specially against larger perturbations in the CIFAR-10 dataset, which intuitively is pleasing:
as larger perturbations are naturally more distinguishable, the detection capability of the network is
successfully leveraged for improving the adversarial robustness. Let us now take a closer look at the
performance by focusing on the detection capability.

Effectiveness of the detection class. By nature, the proposed classiﬁcation “adaptively chooses”
between (robust) correct classiﬁcation and detection of adversarial or difﬁcult inputs during the
training. This gives rise to two phenomena:

(1) In veriﬁably robust methods, natural image accuracy declines as robustness improves. In the pro-
posed approach however, a considerable number of misclassiﬁed natural inputs are in fact abstained
on, which in certain applications is more favorable than assigning them to a wrong class, as classi-
ﬁers without detection capability would: compare 30.5% abstain and 25.6% ‘wrong-class misclassi-
ﬁcation’ (other than abstain and the correct class) in IBP-with-detection, with that of 53.7% ‘wrong-
class’ misclassiﬁcation in IBP on natural CIFAR-10 images in networks trained for (cid:15) = 8/255.

(2) Regardless of the training procedure, the proposed classiﬁer with detection can still be veriﬁed
using veriﬁcation in Eq. 4 to obtain its guaranteed robustness with only considering the correct
class. Thus, comparing this veriﬁcation percentage with that of Eq. 11 highlights the effectiveness
of the abstain class in detecting perturbed images and increasing robustness: for instance, using
our method 76.07% maximum robust error successfully decreases to 63.63% by considering the
detection capability, on CIFAR-10 trained for (cid:15) = 8/255, compared to 69.92% in IBP without
detection.

7

Published as a conference paper at ICLR 2021

dataset

attack

MNIST

(cid:15)test = 0.3
(cid:15)train = 0.4

(cid:15)test = 0.4
(cid:15)train = 0.4

(cid:15)test = 2/255
(cid:15)train = 2.2/255

CIFAR-10

(cid:15)test = 8/255
(cid:15)train = 8.8/255

method
IBP
IBP w/ detection
Best recorded in literature
IBP (Gowal et al., 2018)
IBP-CROWN (Zhang et al., 2020)
Xiao et al. (2018)
Mirman et al. (2019)
Balunovic & Vechev (2020)
Wong & Kolter (2018)

IBP
IBP w/ detection
Best recorded in literature
IBP (Gowal et al., 2018)
IBP-CROWN (Zhang et al., 2020)

IBP
IBP w/ detection
Best recorded in literature
IBP (Gowal et al., 2018)
IBP-CROWN (Zhang et al., 2020)
Balunovic & Vechev (2020)
Mirman et al. (2018)
Wong & Kolter (2018)
Xiao et al. (2018)

IBP
IBP w/ detection
Best recorded in literature
IBP (Gowal et al., 2018)
IBP-CROWN (Zhang et al., 2020)
Mirman et al. (2019)
Balunovic & Vechev (2020)
Xiao et al. (2018)
Wong & Kolter (2018)

(cid:15)test = 16/255
(cid:15)train = 16.7/255

IBP
IBP w/ detection
Best recorded in literature

IBP-CROWN (Zhang et al., 2020)

standard err
2.12
4.34

veriﬁed err
8.47
5.98

pgd-attack-success
6.78
4.15

1.66
1.82
2.67
2.8
2.7
14.87
2.74
4.79

1.66
2.17

38.54
34.66

29.84
28.48
21.6
38.0
31.72
38.88
53.69
55.60

50.51
54.02
43.8
48.3
59.55
71.33
68.97
66.37

66.06

8.21
7.02
19.32
11.2
14.3
43.10
14.80
11.29

15.01
12.06

55.21
57.9

55.88
46.03
39.5
47.8
46.11
54.07
69.92
63.63

68.44+
66.94
72.8
72.5
79.73
78.22
78.12
67.92

76.80

6.12
6.05
7.95
4.6
–
–
11.14
7.55

10.34
9.47

49.72
47.2

49.98
40.28
–
–
–
50.08
65.17
49.22

65.23
65.42
65.3
–
73.22
–
76.66
58.20

75.23

Table 1: The veriﬁed, standard (clean), and PGD attack errors for models trained on MNIST and CIFAR-
10. IBP with detection is to be compared with IBP (without detection capability) to emphasize the suc-
cessful utilization of the detection capability of the network in increasing its veriﬁable as well as empirical
performance. For a more detailed decomposition of the standard and robust error terms see Fig. 1.
+ As reported in Zhang et al. (2020), achieving the 68.44% IBP veriﬁed error requires extra PGD adversarial
training loss, without which the veriﬁed error is 72.91% (LP/MIP veriﬁed) or 73.52% (IBP veriﬁed), thus
our result should be compared to 73.52%.
* Best reported numbers for IBP are computed using mixed integer programming (MIP), which are strictly
smaller than IBP veiﬁed error rates, see table 3 and 4 in Gowal et al. (2018). For fair comparison, we report
IBP veriﬁed error rates from table 3 therein.
** Best reported results from the literature may use different network architectures, and empirical PGD
error rate may have been computed under different settings, e.g., number of steps and restarts.
*** Number in the IBP rows in this table are the best between (Zhang et al., 2020) and our experiments,
while results from (Gowal et al., 2018) are reported under best literature record for IBP.
† It is important to note that unlike robust classiﬁcation, the proposed joint classiﬁcation/detection does
successfully leverage the detection capability to decrease the veriﬁed error rate by rejecting some adversar-
ial examples, which makes direct comparison of these values difﬁcult. However since there exists no other
veriﬁable detection scheme, such comparison is made here to show the effect of successful detection; see
Figure 1 for a detailed discussion on this.

See Fig. 1 for decomposition of the performance metrics of the proposed network over CIFAR-10
dataset, demonstrating the effectiveness of the abstain class in detecting “difﬁcult” natural images
while also increasing the robustness certiﬁcate over adversarial inputs.

5.3 NATURAL VERSUS ADVERSARIAL ERROR TRADEOFF

Reporting a single set point in the Pareto Frontier as reported in Table 1 gives limited understanding
on how different methods trade off natural versus robust error. To address this, a more detailed study
on this trade-off in IBP-based robust classiﬁcation with and without detection is discissed here.

In order to get the best performance for IBP-based robust training without detection (that is λ1 = 0),
and since it is not known whether varying κend or λ2 will lead to a better performance, we have

8

Published as a conference paper at ICLR 2021

(a) Accuracy on natural (clean) images

(b) Veriﬁed accuracy on adversarial images

Figure 1: Decomposition of accuracy and veriﬁed accuracy on CIFAR-10 dataste: the detection
capability of the network can increase robustness by adaptively abstaining on adversarial inputs
while also abstaining on some natural images rather than misclassifying them.

(a) CIFAR-10, (cid:15) = 8/255

(b) CIFAR-10, (cid:15) = 12/255

(c) CIFAR-10, (cid:15) = 16/255

Figure 2: Naural versus robust error tradeoff for IBP (λ1 = 0) and IBP-with-detection (λ1 > 0)
on CIFAR-10 dataset for various perturbation sizes (cid:15) = 8/255, 12/255, 16/255. Lower curve is
better. IBP-with-detection is effectively utilizing its detection capability to adaptively trade natural
and robust performance, leading to improved certiﬁed robustness against adversarial perturbations.

trained the classiﬁcation networks in two ways: (1) setting κend = 0.5 and varying λ2 ∈ [0, 3], and
(2) setting λ2 = 1 and varying κend ∈ [0, 0.5], to get multiple set points along the Pareto Frontier.

Similarly, for IBP-with-detection-based classiﬁcation, we have set κend = 0.5 , λ1 = 0.6, 0.8, 1.0 for
MNIST and λ1 = 1.0 for CIFAR-10, and varied λ2 ∈ [1 4] to get various points along the frontier.
The network is trained for various (cid:15) values, with other training parameters as stated in Appendix B.

Results are plotted in Fig. 2 and 3 (presented in the Appendix due to space limitation). As shown,
the classiﬁer enhanced with detection capability is better able to trade natural and robust accuracy,
thus attaining higher robustness by trading small decrease in natural accuracy. This together with
the fact that the natural accuracy decrease is also partly handled by abstaining of such natural im-
ages that would have been misclassiﬁed (as one of the original K classes) otherwise, demonstrates
the effective utilization of the detection capability in the proposed method. It is important to note
that IBP w/detection allows us to obtain additional regions on this Pareto frontier that traditional-
robust-classiﬁers without detection cannot obtain, and could potentially provide additional gain to
what is achievable by other various improvement techniques such as tighter relaxation and bound
propagation methods.

6 CONCLUSION

We proposed a new method for jointly training a provably robust classiﬁer and detector. By intro-
ducing an additional “abstain/detection” into a classiﬁer, we have proposed a veriﬁcation scheme for
classiﬁers with detection under adversarial settings, and shown that such networks can be efﬁciently
trained be extending the common IBP relaxation techniques. The effectiveness of the proposed de-
tection scheme with provable guarantees versus SOTA robust veriﬁable classiﬁcation methods is
corroborated by empirical tests on MNIST and CIFAR-10, specially against large perturbations.

9

IBPIBP w\ Detection=8/2550102030405060708090IBPIBP w\ Detection=16/2550102030405060708090correct class.abstainmisclassificationIBPIBP w\ Detection=8/2550102030405060708090IBPIBP w\ Detection=16/2550102030405060708090robust class.robust class. w\ detectionnon-verified0.450.500.550.600.65natural error0.450.500.550.600.650.700.75robust error0.550.600.650.70natural error0.450.500.550.600.650.700.750.80robust errorIBP (1=0), varying 2IBP (1=0), varying endIBP w/ detection (1=1), varying 20.600.620.640.660.68natural error0.600.650.700.750.800.85robust errorPublished as a conference paper at ICLR 2021

REFERENCES

Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of
security: Circumventing defenses to adversarial examples. Proceedings of Machine Learning
Research, pp. 274–283. PMLR, 10–15 Jul 2018.

Mitali Bafna, Jack Murtagh, and Nikhil Vyas. Thwarting adversarial examples: An l0-robust sparse
In Advances in Neural Information Processing Systems, pp. 10075–10085,

Fourier transform.
2018.

Mislav Balunovic and Martin Vechev. Adversarial training and provable defenses: Bridging the gap.

In International Conference on Learning Representations, 2020.

Julian Bitterwolf, Alexander Meinke, and Matthias Hein. Certiﬁably adversarially robust detection
of out-of-distribution data. In Advances in Neural Information Processing Systems, volume 33,
pp. 16085–16095, 2020.

Krishnamurthy Dvijotham, Robert Stanforth, Sven Gowal, Timothy A Mann, and Pushmeet Kohli.
A dual approach to scalable veriﬁcation of deep networks. In UAI, volume 1, pp. 550–559, 2018.

Ruediger Ehlers. Formal veriﬁcation of piece-wise linear feed-forward neural networks. In Interna-
tional Symposium on Automated Technology for Veriﬁcation and Analysis, pp. 269–286. Springer,
2017.

Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George Pappas. Efﬁcient
and accurate estimation of lipschitz constants for deep neural networks. In Advances in Neural
Information Processing Systems 32, pp. 11427–11438. 2019.

Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Ue-
sato, Relja Arandjelovic, Timothy Mann, and Pushmeet Kohli. On the effectiveness of interval
bound propagation for training veriﬁably robust models. arXiv preprint arXiv:1810.12715, 2018.

Cassidy Laidlaw and Soheil Feizi. Playing it safe: Adversarial robustness with an abstain option.

arXiv preprint arXiv:1911.11253, 2019.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.

Towards deep learning models resistant to adversarial attacks. ICLR, 2017.

Matthew Mirman, Timon Gehr, and Martin Vechev. Differentiable abstract interpretation for prov-
ably robust neural networks. In International Conference on Machine Learning, pp. 3578–3586,
2018.

Matthew Mirman, Gagandeep Singh, and Martin Vechev. A provable defense for deep residual

networks. arXiv preprint arXiv:1903.12519, 2019.

Tianyu Pang, Kun Xu, Chao Du, Ning Chen, and Jun Zhu. Improving adversarial robustness via
promoting ensemble diversity. In International Conference on Machine Learning, pp. 4970–4979,
2019a.

Tianyu Pang, Kun Xu, and Jun Zhu. Mixup inference: Better exploiting mixup to defend adversarial

attacks. In International Conference on Learning Representations, 2019b.

Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Certiﬁed defenses against adversarial exam-

ples. ICLR, 2018.

Kevin Roth, Yannic Kilcher, and Thomas Hofmann. The odds are odd: A statistical test for detecting
adversarial examples. In International Conference on Machine Learning, pp. 5498–5507, 2019.

Hadi Salman, Greg Yang, Huan Zhang, Cho-Jui Hsieh, and Pengchuan Zhang. A convex relaxation
In Advances in Neural Information

barrier to tight robustness veriﬁcation of neural networks.
Processing Systems 32, pp. 9835–9846. 2019.

Ali Shafahi, Mahyar Najibi, Mohammad Amin Ghiasi, Zheng Xu, John Dickerson, Christoph
Studer, Larry S Davis, Gavin Taylor, and Tom Goldstein. Adversarial training for free!
In
Advances in Neural Information Processing Systems, pp. 3358–3369, 2019.

10

Published as a conference paper at ICLR 2021

Fatemeh Sheikholeslami, Swayambhoo Jain, and Georgios B Giannakis. Minimum uncertainty
based detection of adversaries in deep neural networks. In Information Theory and Applications
Workshop (ITA). IEEE, 2020.

Lewis Smith and Yarin Gal. Understanding measures of uncertainty for adversarial example detec-

tion. arXiv preprint arXiv:1803.08533, 2018.

David Stutz, Matthias Hein, and Bernt Schiele. Conﬁdence-calibrated adversarial training: Gener-

alizing to unseen attacks. In International Conference on Machine Learning, 2020.

Vincent Tjeng, Kai Xiao, and Russ Tedrake. Evaluating robustness of neural networks with mixed

integer programming. ICLR, 2017.

Florian Tramer, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to

adversarial example defenses. arXiv preprint arXiv:2002.08347, 2020.

Gunjan Verma and Ananthram Swami. Error correcting output codes improve probability estimation
and adversarial robustness of deep neural networks. In Advances in Neural Information Process-
ing Systems, pp. 8643–8653, 2019.

Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer
adversarial polytope. volume 80 of Proceedings of Machine Learning Research, pp. 5286–5295.
PMLR, 10–15 Jul 2018.

Eric Wong, Leslie Rice, and J Zico Kolter. Fast is better than free: Revisiting adversarial training.

In International Conference on Learning Representations, 2019.

Chang Xiao, Peilin Zhong, and Changxi Zheng. Resisting adversarial attacks by k-winners-take-all.

arXiv preprint arXiv:1905.10510, 2019.

Kai Y Xiao, Vincent Tjeng, Nur Muhammad Mahi Shaﬁullah, and Aleksander Madry. Training for
faster adversarial robustness veriﬁcation via inducing relu stability. In International Conference
on Learning Representations, 2018.

Yuzhe Yang, Guo Zhang, Dina Katabi, and Zhi Xu. Me-net: Towards effective adversarial robustness
with matrix estimation. In International Conference on Machine Learning, pp. 7025–7034, 2019.

Xuwang Yin, Soheil Kolouri, and Gustavo K Rohde. Gat: Generative adversarial training for ad-
versarial example detection and robust classiﬁcation. In International Conference on Learning
Representations, 2020.

Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael I Jordan.

Theoretically principled trade-off between robustness and accuracy. In ICML, 2019.

Huan Zhang, Hongge Chen, Chaowei Xiao, Sven Gowal, Robert Stanforth, Bo Li, Duane Boning,
and Cho-Jui Hsieh. Towards stable and efﬁcient training of veriﬁably robust neural networks. In
International Conference on Learning Representations, 2020.

11

Published as a conference paper at ICLR 2021

A APPENDIX

A.1 PROOF OF THEOREM 1

Since ZL ∈ ˆZL it trivially holds that

min
z∈ ˆZL

max{c(cid:62)

y,iz, c(cid:62)

a,iz} ≤ min
z∈ZL

max{c(cid:62)

y,iz, c(cid:62)

a,iz}

(20)

The lower bound is now a convex minimization, which can be rewritten as

min
z∈ ˆZL

max{c(cid:62)

y,iz, c(cid:62)

a,iz} = min
τ,z∈ ˆZL

τ

s. t. c(cid:62)

y,iz ≤ τ

, c(cid:62)

a,iz ≤ τ.

Deﬁning the slack variables ηa ≥ 0 and ηy ≥ 0 for the inequality constraints, the Lagrangian can be
written as

L(τ, z, ηa, ηy) = τ + ηa(c(cid:62)

a,iz − τ ) + ηy(c(cid:62)

y,iz − τ )

and minimizing L(τ, z, ηa, ηy) with respect to the primal variable τ , yields ηa + ηy = 1. Deﬁning
η := ηa = 1 − ηy, and using the fact that the dual maximization always serves as a lower bound on
the primal we get

max
0≤η≤1

min
z∈ ˆZL

(cid:16)

η ca,i + (1 − η) cy,i

(cid:17)(cid:62)

z ≤ min
z∈ZL

max{c(cid:62)

y,iz, c(cid:62)

a,iz}.(cid:3)

A.2 PROOF OF THEOREM 2

Following on the statement of Theorem 1 and by substituting z = W(cid:62)

L zL−1 + bL, we get

max
0≤η≤1

min
zL−1∈ ˆZL−1

(cid:16)

η ca,i+(1−η) cy,i

(cid:17)(cid:62)(cid:16)

W(cid:62)

L zL−1+bL

(cid:17)

≤ min
z∈ ˆZL

which can be reordered as

max{c(cid:62)

y,iz, c(cid:62)

a,iz} (21)

max
0≤η≤1

min
zL−1≤zL−1≤¯zL−1

(ω1 + ηω2)(cid:62)zL−1 + η ω3 + ω4

(22)

where ω1 := WLcy,i, ω2 := WL(ca,i − cy,i), ω3 := b(cid:62)
then equals

L (ca,i − cy,i), and ω4 := b(cid:62)

L cy,i, which

max
0≤η≤1

(ω1 + ηω2)(cid:62)ˆzL−1 + η ω3 + ω4

(23)

where minimization w.r.t. zL−1 is solved by the (this is under the setting for most networks with
positive activations, and thus lower bound zl is always non-negative)

[ˆzL−1]j =

(cid:26)[¯zL−1]j
[zL−1]j

if
if

[ω1 + ηω2]j ≤ 0
[ω1 + ηω2]j ≥ 0

(24)

and can be rewritten as

max
0≤η≤1

nL−1
(cid:88)

(cid:104)

j=1

ω1 +ηω2

(cid:105)

j

and can be rewritten as

(cid:32)

(cid:33)

1{ω1,j +ηω2,j ≤0}[¯zL−1]j +1{ω1,j +ηω2,j ≥0}[zL−1]j

+η ω3 +ω4 (25)

max
0≤η≤1

nL−1
(cid:88)

(cid:32)

(cid:33)

1{ω1,j +ηω2,j ≤0}[ω1 ◦ ¯zL−1 + ηω2 ◦ ¯zL−1]j + 1{ω1,j +ηω2,j ≥0}[ω1 ◦ zL−1 + ω2 ◦ zL−1]j

j=1
+ η ω3 + ω4

(26)

where “◦” denotes the elementwise multiplication. Thus, due to the concavity of the dual, optimal
η can be found by evaluationg the objective in between the break points which are given by ζ :=
[ζ1, ..., ζnL−1 ] with its j-th element deﬁned as ζj := −ω1,j/ω2,j.

12

Published as a conference paper at ICLR 2021

To do this, let us use s to denote the nL-ary tuple of indices that sorts ζ. That is

˜ζ = [˜ζ1, ..., ˜ζnL−1 ] := Πs(ζ) := [ζs1, ..., ζsnL−1

]

s.t.

ζs1 ≤ ... ≤ ζsnL−1

with operator Πs(.) denoting the permutation of its arguments according to s, such that ˜ζi = ζsi∀i,
and ˜ζ is sorted in the increasing order .

We can also rewrite the problem by summing over the indices in the sorting set s instead, as

max
0≤η≤1

nL−1
(cid:88)

(cid:32)

(cid:33)

1{ω1,j +ηω2,j ≤0}[ω1 ◦ ¯zL−1 + ηω2 ◦ ¯zL−1]sj + 1{ω1,j +ηω2,j ≥0}[ω1 ◦ zL−1 + ω2 ◦ zL−1]sj

j=1
+ η ω3 + ω4.

(27)

Now let us deﬁne u1 := Πs(ω1 ◦ zL−1) , ¯u1 := Πs(ω1 ◦ ¯zL−1), u2 := Πs(ω2 ◦ zL−1) , ¯u2 :=
Πs(ω2 ◦ ¯zL−1), we get
nL−1
(cid:88)

(cid:32)

max
0≤η≤1

1(cid:110)

j=1

{η≤˜ζj and ω2,sj >0} or {η≥˜ζj and ω2,sj <0}

(cid:111) (¯u1,j + η¯u2,j)

+ 1(cid:110)

{η≥˜ζj and ω2,sj >0} or {η≤˜ζj and ω2,sj <0}

(cid:33)

(cid:0)u1,j + ηu2,j

(cid:1)

(cid:111)

+ η ω3 + ω4.

(28)

In order to break the objective of maximization into piece-wise linear programming subproblems,
let us ﬁrst identify the (indices of) ζsi values that fall in the feasible set 0 ≤ η ≤ 1 by

m = min
ζsν ≥0

ν

and M = max
ζsν ≤1

ν

The overall maximization can now be reduced to piece-wise subproblems over sets ˜ζν ≤ η ≤ ˜ζν+1
for m − 1 ≤ ν ≤ M as

max
max{0,˜ζν }≤η≤min{1,˜ζν+1}

nL−1
(cid:88)

j=1

(cid:32)

1(cid:110)

{η≤˜ζj and ω2,sj >0} or {η≥˜ζj and ω2,sj <0}

(cid:111) (¯u1,j + η¯u2,j)

+ 1(cid:110)

{η≥˜ζj and ω2,sj >0} or {η≤˜ζj and ω2,sj <0}

+ η ω3 + ω4.

(cid:33)

(cid:0)u1,j + ηu2,j

(cid:1)

(cid:111)

(29)

Since each of these subproblems are maximized at the boundaries of the feasible sets, the overall
maximization essentially reduces to evaluation of the following objective function at (M − m + 3)
points η = 0, ˜ζm, ˜ζm+1, · · · , ˜ζM −1, ˜ζM , 1

g(η) =

nL−1
(cid:88)

j=1

(cid:32)

1{ω1,j +ηω2,j ≤0} (¯u1,j + η¯u2,j) + 1{ω1,j +ηω2,j ≥0}

(cid:33)

(cid:0)u1,j + ηu2,j

(cid:1)

+ η ω3 + ω4

Values of g(η) can be efﬁciently computed by a forward cumulative sum and forward-backward
cumulative sum of u1 and u2, ¯u1 and ¯u2, thus imposing the overall complexity which is dominated
.(cid:3)
by the sorting at O(nL−1 log(nL−1)) in an efﬁcient implementation.

A.3 DESCRIPTION OF ALGORITHM 1

Here is a step-by-step walk-through for Algorithm 1, with insight on how these steps are performed.

1. Form vectors ω1 and ω2, which are the last layer values as ω1 = WLcy,i, ω2 =
L (ca,i − cy,i), and ω4 := b(cid:62)

WL(ca,i − cy,i) , ω3 := b(cid:62)

L cy,i.

13

Published as a conference paper at ICLR 2021

2. Deﬁne ζ = [ζ1, ..., ζnL] := −ω1/ω2 and get the vector of indices s that sorts it, i.e.,

ζs1 ≤ · · · ≤ ζsnL−1

3. Form the element-wise product of (ω1, ω2) with (zL−1, ¯zL−1)), and sort them according

to the index set s.
u1 = Πs(ω1 ◦ zL−1) , ¯u1 = Πs(ω1 ◦ ¯zL−1), u2 := Πs(ω2 ◦ zL−1) , ¯u2 := Πs(ω2 ◦ ¯zL−1).
4. Get the lowest and highest indexes (m, M ) such that the sorted ζ vector value at those

indices are in the feasible set, between 0 and 1.

5. Iterate over the feasible values of η = 0, ζsm , ζsm+1, · · · , ζsM −1, ζsM , 1 and compute the

corresponding objective values

nL−1
(cid:88)

(cid:16)

g(η) =

1{ω1,j +ηω2,j ≤0} (¯u1,j + η¯u2,j) + 1{ω1,j +ηω2,j ≥0}

(cid:0)u1,j + ηu2,j

(cid:1) (cid:17)

j=1
+ η ω3 + ω4

6. Return the maximum value of g(η) over the evaluated points.

A.4 PROOF OF THEOREM 3

Let us start by splitting the feasible set into disjoint sets of

ˆZ a

L−1 := {zL−1 | zL−1,a ≥ zL−1,y}, and ˆZ y

L−1 := {zL−1 | zL−1,a < zL−1,y}

where

Proof is carried out by considering z ∈ ˆZ y

ˆZL−1 = ˆZ y

L−1 ∪ ˆZ a

L−1, and ˆZ y
L−1 and z ∈ ˆZ a

L−1 = ∅.

L−1 ∩ ˆZ a
L−1, separately.

Restricting z ∈ ˆZ y

L−1 we have (cid:96)xent\a(fθ(x + δ), y) ≤ (cid:96)xent\y(fθ(x + δ), a) which leads to

Labstain

robust (x, y; θ) = max
δ∈∆

min

(cid:110)

(cid:96)xent\a(fθ(x + δ), y), (cid:96)xent\y(fθ(x + δ), a)

(cid:111)

≤ max
zL−1∈ ˆZ y

L−1

(cid:96)xent\a(zL, y)

s.t. zL = W(cid:62)

L zL−1 + bL

(30)

(31)

Loss function (cid:96)xent\a is the cross entropy loss deﬁned on the K-dimensional vector [zL,1, · · · , zL,K]
and class y, and thus following Wong & Kolter (2018) given its transnational invariance equals

max
zL−1∈ ˆZ y

L−1

(cid:96)xent\a(zL, y) = max
zL−1∈ ˆZ y

L−1

(cid:96)xent\a(zL − zL,y1, y)

s.t. zL = W(cid:62)

L zL−1 + bL (32)

with 1 denoting the (K + 1)-dimensional vector of all ones. Given the invariance of (cid:96)xent\a with
respect to zL,a, it can ﬁnally be upperbounded by taking the upperbound for all i indices where
i = 1, ..., K, i (cid:54)= a, y and lowerbound at index i = y. Note that for i = y, value [zL − zL,y1]i = 0,
and a lower bound on other entries i = 1, ..., K, i (cid:54)= a, y can be obtained by

zL,i − zL,y = − max{zL,y − zL,i, zL,a − zL,i} = − max{c(cid:62)
max{c(cid:62)

y,iz, c(cid:62)
η,¯η
a,iz} ≤ −Ji(x, y) ≤ −J
i

y,iz, c(cid:62)

a,iz}

(x, y)

(33)

(34)

≤ − min
zL∈ZL
where the ﬁrst equality holds since ˆZ y
inequality is due to Theorem 2, and third inequality is given by Eq. 15.
Thus, for z ∈ ˆZ y

L−1 the loss term is now upperbounded by

L−1 := {zL−1 | zL−1,a < zL−1,y} for z ∈ ˆZ y

L−1, second

where

Labstain

robust (x, y; θ) ≤ (cid:96)xent\a(−J(cid:15),θ(x, y), y)

[J(cid:15),θ(x, y)]i =

(cid:40)
0
J

if
(x, y)

η,¯η
i

i = a, y

otherwise.

14

(35)

Published as a conference paper at ICLR 2021

Network layers
Conv 64 3 × 3 + 1
Conv 64 3 × 3 + 1
Conv 128 3 × 3 + 2
Conv 128 3 × 3 + 1
Conv 128 3 × 3 + 1
Fully Conn. 512
# hidden
# params.

230K
17M

Table 2: Network architecture. Similar to the Large network used in (Gowal et al., 2018)

Similarly, it can be shown that for Thus, for z ∈ ˆZ a

L−1 the loss term is now upperbounded by

Labstain

robust (x, y; θ) ≤ (cid:96)xent\y(−J(cid:15),θ(x, y), a).

The equality of (cid:96)xent\y(−J(cid:15),θ(x, y), a) = (cid:96)xent\a(−J(cid:15),θ(x, y), y) trivially follows from the fact that
[J(cid:15),θ(x, y)]i = 0 for i = a, y.
Thus, since ˆZL−1 = ˆZ y

L−1, the proof is complete.

L−1 ∪ ˆZ a

.(cid:3)

B APPENDIX: EXPERIMENT SET UP

Training parameters and schedules are similar to (Gowal et al., 2018) and (Zhang et al., 2020), and
outlined in detail here. For training the classiﬁer network with architecture given in Table 2, for
both datasets, Adam optimizer with learning rate of 5 × 10−4 is used. Unless stated differently, κ
is scheduled by a linear ramp-down process, starting at 1, which after a warm-up perio,d is ramped
down to value κend = 0.5. Value of (cid:15) during the training is also simultaneously scheduled by a linear
ramp-up, starting at 0, and ramped up to the ﬁnal value of (cid:15)train, reported in Tabel 1, and networks
are trained with a single NVIDIA Tesla V100S GPU.

• For MNIST, the network is trained in 100 epochs with batchsize of 100 (total of 60K steps).
A warm up period of 3 epochs (2K steps) is used (normal classiﬁcation training with no
robust loss), followed up by a ramp-up duration of 18 epochs (10K steps), and the learning
rate is decayed ×10 at epochs 25 and 42. No data augmentation is used. Furthermore,
ﬁxed selection of ¯η = 0.9 and η = 0.1 during training is used for this dataset with no
ramp-down. Reported numbers in Table 1 corresponds to λ1 = 1 and λ2 = 2 for (cid:15) = 0.3,
and λ1 = 0.6 and λ2 = 1 for (cid:15) = 0.4 respectively.

• For CIFAR10, the network is trained in 3200 epochs with batchsize of 1600 (total of 100K
steps). A warm up period of 320 epochs (10K steps) is used (normal classiﬁcation training
with no robust loss), followed up by a ramp-up duration of 1600 epochs (50K steps), and
the learning rate is decayed ×10 at epochs 2600 and 3040 (60k and 90K steps). Random
translations and ﬂips, and normalization of each image channel (using the channel statistics
from the train set) is used during training. Furthermore, during training for all (cid:15) values we
= 0.1 is used during training,
have selected ¯ηstart = 1.0 and ¯ηend = 0.9. Additionally, η
with η
= 0.4
for (cid:15) = 12/255, and η
= 0.5 for (cid:15) = 16/255. The intuition behind these parameters
selection lies in Remark 2, as large η values promote the abstain option more, so for large
(cid:15), we start with larger η
as well. Reported numbers in Tabel 1 correspond to λ1 = 1
for all (cid:15) values, and λ2 = 3.0 for (cid:15) = 2/255, λ2 = 2.9 for (cid:15) = 8/255, and λ2 = 3.1 for
(cid:15) = 16/255 to insure similar natural accuracy for fair comparison against other methods.

= 0.1 for (cid:15) = 2/255 (no ramp down), η

end
= 0.3 for (cid:15) = 8/255, η

start

start

start

start

start

15

Published as a conference paper at ICLR 2021

(a) MNIST (cid:15) = 0.3

(b) MNIST (cid:15) = 0.4

Figure 3: Naural versus robust error tradeoff for IBP (λ1 = 0) and IBP-with-detection (λ1 > 0) on
MNIST dataset for various perturbation sizes (cid:15) = 0.3 and (cid:15) = 0.4. Points closer to the origin are
better. IBP-with-detection is effectively utilizing its detection capability to adaptively trade natural
and robust performance, leading to improved certiﬁed robustness against adversarial perturbations.

B.1 PARETO FRONTIER FOR MNIST DATASET

B.2 EMPIRICAL ATTACK SUCCESS RATE USING PGD ATTACKS

In order to obtain empirical attack success on the trained networks, adversarial perturbations are
sought by solving

(cid:16)

max
δ∈∆(cid:15)

max
i(cid:54)=a,y

zL,i − max{zL,y, zL,a}

(cid:17)

(36)

This attack is indeed an adaptive attack as it aims at circumventing the detection while trying to cause
misclassiﬁcation (Tramer et al., 2020). Perturbations are sought by maximizing this objective using
PGD with 200-steps for mnist and 500-steps for CIFAR-10 Madry et al. (2017), with 10 random
restarts. It is interesting to note that the achieved attack success rate in Table 1 is well below the
veriﬁed robust error, further implying the effectiveness of incorporation of the detection mechanism
as the true robustness of the system against practical adaptive PGD attacks are considerably stronger
in comparison to robust classiﬁcation without detection.

16

