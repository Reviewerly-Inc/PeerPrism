Published as a conference paper at ICLR 2021

DECENTRALIZED ATTRIBUTION
OF GENERATIVE MODELS

Changhoon Kim1,∗, Yi Ren2,∗, Yezhou Yang1
School of Computing, Informatics, and Decision Systems Engineering1
School for Engineering of Matter, Transport, and Energy2
Arizona State University
{kch, yiren, yz.yang}@asu.edu

ABSTRACT

Growing applications of generative models have led to new threats such as mali-
cious personation and digital copyright infringement. One solution to these threats
is model attribution, i.e., the identiﬁcation of user-end models where the contents
under question are generated from. Existing studies showed empirical feasibility
of attribution through a centralized classiﬁer trained on all user-end models. How-
ever, this approach is not scalable in reality as the number of models ever grows.
Neither does it provide an attributability guarantee. To this end, this paper studies
decentralized attribution, which relies on binary classiﬁers associated with each
user-end model. Each binary classiﬁer is parameterized by a user-speciﬁc key and
distinguishes its associated model distribution from the authentic data distribution.
We develop sufﬁcient conditions of the keys that guarantee an attributability lower
bound. Our method is validated on MNIST, CelebA, and FFHQ datasets. We also
examine the trade-off between generation quality and robustness of attribution
against adversarial post-processes.1

INTRODUCTION

1
Recent advances in generative models (Good-
fellow et al., 2014) have enabled the creation
of synthetic contents that are indistinguish-
able even by naked eyes (Pathak et al., 2016;
Zhu et al., 2017; Zhang et al., 2017; Kar-
ras et al., 2017; Wang et al., 2018; Brock
et al., 2018; Miyato et al., 2018; Choi et al.,
2018; Karras et al., 2019a;b; Choi et al., 2019).
Such successes raised serious concerns regard-
ing emerging threats due to the applications
of generative models (Kelly, 2019; Breland,
2019). This paper is concerned about two
particular types of threats, namely, malicious
personation (Satter, 2019) , and digital copy-
right infringement. In the former, the attacker
uses generative models to create and dissem-
inate inappropriate or illegal contents; in the
latter, the attacker steals the ownership of a
copyrighted content (e.g., an art piece created
through the assistance of a generative model)
by making modiﬁcations to it.

Figure 1: FFHQ dataset projected to the space
spanned by two keys φ1 and φ2. We develop suf-
ﬁcient conditions for model attribution: Perturb-
ing the authentic dataset along different keys with
mutual angles larger than a data-dependent thresh-
old guarantees attributability of the perturbed dis-
(a) A threshold of 90 deg sufﬁces for
tributions.
benchmark datasets (MNIST, CelebA, FFHQ). (b)
Smaller angles may not guarantee attributability.

We study model attribution, a solution that may address both threats. Model attribution is deﬁned as
the identiﬁcation of user-end models where the contents under question are generated from. Existing

∗Equal contribution.
1https://github.com/ASU-Active-Perception-Group/decentralized_

attribution_of_generative_models

1

Published as a conference paper at ICLR 2021

studies demonstrated empirical feasibility of attribution through a centralized classiﬁer trained on
all existing user-end models (Yu et al., 2018). However, this approach is not scalable in reality
where the number of models ever grows. Neither does it provide an attributability guarantee. To this
end, we propose in this paper a decentralized attribution scheme: Instead of a centralized classiﬁer,
we use a set of binary linear classiﬁers associated with each user-end model. Each classiﬁer is
parameterized by a user-speciﬁc key and distinguishes its associated model distribution from the
authentic data distribution. For correct attribution, we expect one-hot classiﬁcation outcomes for
generated contents, and a zero vector for authentic data. To achieve correct attribution, we study
the sufﬁcient conditions of the user-speciﬁc keys that guarantee an attributability lower bound. The
resultant conditions are used to develop an algorithm for computing the keys. Lastly, we assume
that attackers can post-process generated contents to potentially deny the attribution, and study the
tradeoff between generation quality and robustness of attribution against post-processes.

Problem formulation We assume that for a given dataset D ⊂ Rdx, the registry generates user-
speciﬁc keys, Φ := {φ1, φ2, ...} where φi ∈ Rdx and ||φi|| = 1. || · || is the l2 norm. A user-end
generative model is denoted by Gφ(·; θ) : Rdz → Rdx where z and x are the latent and output
variables, respectively, and θ are the model parameters. When necessary, we will suppress θ and
φ to reduce the notational burden. The dissemination of the user-end models is accompanied by a
public service that tells whether a query content belongs to Gφ (labeled as 1) or not (labeled as −1).
We model the underlying binary linear classiﬁer as fφ(x) = sign(φT x). Note that linear models are
necessary for the development of sufﬁcient conditions of attribution presented in this paper, although
sufﬁcient conditions for nonlinear classiﬁers are worth exploring in the future.

The following quantities are central to our investigation: (1) Distinguishability of Gφ measures the
accuracy of fφ(x) at classifying Gφ against D:

D(Gφ) :=

1
2

Ex∼PGφ ,x0∼PD [1(fφ(x) = 1) + 1(fφ(x0) = −1)] .

(1)

Here PD is the authentic data distribution, and PGφ the user-end distribution dependent on φ. G
is (1 − δ)-distinguishable for some δ ∈ (0, 1] when D(G) ≥ 1 − δ. (2) Attributability measures
the averaged multi-class classiﬁcation accuracy of each model distribution over the collection G :=
{Gφ1, ..., GφN }:

A(G) :=

1
N

N
(cid:88)

i=1

Ex∼Gφi

1(φT

j x < 0, ∀ j (cid:54)= i, φT

i x > 0).

(2)

G is (1 − δ)-attributable when A(G) ≥ 1 − δ. (3) Lastly, We denote by G(·; θ0) (or shortened as
G0) the root model trained on D, and assume PG0 = PD. We will measure the (lack of) generation
quality of Gφ by the FID score (Heusel et al., 2017) and the l2 norm of the mean output perturbation:
∆x(φ) = Ez∼Pz [Gφ(z; θ) − G(z; θ0)],
(3)

where Pz is the latent distribution.

This paper investigates the following question: What are the sufﬁcient conditions of keys so that
the user-end generative models can achieve distinguishability individually and attributability col-
lectively, while maintaining their generation quality?

Contributions We claim the following contributions:

1. We develop sufﬁcient conditions of keys for distinguishability and attributability, which
connect these metrics with the geometry of the data distribution, the angles between keys,
and the generation quality.

2. The sufﬁcient conditions lead to simple design rules for the keys: keys should be (1) data
compliant, i.e., φT x < 0 for x ∼ PD, and (2) orthogonal to each other. We validate
these rules using DCGAN (Radford et al., 2015) and StyleGAN (Karras et al., 2019a)
on benchmark datasets including MNIST (LeCun & Cortes, 2010), CelebA (Liu et al.,
2015), and FFHQ (Karras et al., 2019a). See Fig. 1 for a visualization of the attributable
distributions perturbed from the authentic FFHQ dataset.

3. We empirically test the tradeoff between generation quality and robust attributability under
random post-processes including image blurring, cropping, noising, JPEG conversion, and
a combination of all.

2

Published as a conference paper at ICLR 2021

2 SUFFICIENT CONDITIONS FOR ATTRIBUTABILITY

From the deﬁnitions (Eq. (1) and Eq. (2)), achieving distinguishability is necessary for attributability.
In the following, we ﬁrst develop the sufﬁcient conditions for distinguishability through Proposition
1 and Theorem 1, and then those for attributability through Theorem 2.

Distinguishability through watermarking First, consider constructing a user-end model Gφ by
simply adding a perturbation ∆x to the outputs of the root model G0. Assuming that φ is data-
compliant, this model can achieve distinguishability by solving the following problem with respect
to ∆x:

min
||∆x||≤ε

Ex∼PD

(cid:2)max{1 − φT (x + ∆x), 0}(cid:3) ,

(4)

where ε > 0 represents a generation quality constraint. The following proposition reveals the con-
nection between distinguishability, data geometry, and generation quality (proof in Appendix A):
Proposition 1. Let dmax(φ) := maxx∼PD |φT x|. If ε ≥ 1 + dmax(φ), then ∆x∗ = (1 + dmax(φ))φ
solves Eq. (4), and fφ(x + ∆x∗) > 0, ∀ x ∼ PD.

Watermarking through retraining user-end models The perturbation ∆x∗ can potentially be
reverse engineered and removed when generative models are white-box to users (e.g., when models
are downloaded by users). Therefore, we propose to instead retrain the user-end models Gφ using
the perturbed dataset Dγ,φ := {G0(z)+γφ | z ∼ Pz} with γ > 0, so that the perturbation is realized
through the model architecture and weights. Speciﬁcally, the retraining ﬁne-tunes G0 so that Gφ(z)
matches with G0(z) + γφ for z ∼ Pz. Since this matching will not be perfect, we use the following
model to characterize the resultant Gφ:

Gφ(z) = G0(z) + γφ + (cid:15),

(5)

where the error (cid:15) ∼ N (µ, Σ). In Sec. 3 we provide statistics of µ and Σ on the benchmark datasets,
to show that the retraining captures the perturbations well (µ close to 0 and small variances in Σ).

Updating Proposition 1 due to the existence of (cid:15) leads to Theorem 1, where we show that γ needs to
be no smaller than dmax(φ) in order for Gφ to achieve distinguishability (proof in Appendix B):
Theorem 1. Let dmax(φ) = maxx∈D |φT x|, σ2(φ) = φT Σφ, δ ∈ [0, 1], and φ be a data-compliant
key. D(Gφ) ≥ 1 − δ/2 if

(cid:115)

γ ≥ dmax(φ) + σ(φ)

log

(cid:19)

(cid:18) 1
δ2

− φT µ.

(6)

Remarks The computation of σ(φ) requires Gφ, which in turn requires γ. Therefore, an iterative
search is needed to determine γ that is small enough to limit the loss of generation quality, yet large
enough for distinguishability (see Alg. 1).

Attributability We can now derive the sufﬁcient conditions for attributability of the generative
models from a set of N keys (proof in Appendix C):
Theorem 2. Let dmin = minx∈D |φT x|, dmax = maxx∈D |φT x|, σ2(φ) = φT Σφ, δ ∈ [0, 1]. Let

a(φ, φ(cid:48)) := −1 +

dmax(φ(cid:48)) + dmin(φ(cid:48)) − 2φ(cid:48)T µ

(cid:113)

σ(φ(cid:48))

log (cid:0) 1
δ2

(cid:1) + dmax(φ(cid:48)) − φ(cid:48)T µ

,

for keys φ and φ(cid:48). Then A(G) ≥ 1 − N δ, if D(G) ≥ 1 − δ for all Gφ ∈ G, and

φT φ(cid:48) ≤ a(φ, φ(cid:48))

for any pair of data-compliant keys φ and φ(cid:48).

(7)

(8)

Remarks When σ(φ(cid:48)) is negligible for all φ(cid:48) and µ = 0, a(φ, φ(cid:48)) is approximately
dmin(φ(cid:48))/dmax(φ(cid:48)) > 0, in which case φT φ(cid:48) ≤ 0 is sufﬁcient for attributability.
In Sec. 3 we
empirically show that this approximation is plausible for the benchmark datasets.

3

Published as a conference paper at ICLR 2021

Figure 2: (a) Validation of Theorem 1: All points should be close to the diagonal line or to its
right. (b) Support for orthogonal keys: Min. RHS value of Eq.(7) for all keys are either positive
(MNIST) or close to zero (CelebA). (c,d) Statistics of µ and Σ for two sample user-end models for
MNIST and CelebA. Small µ and small diag(Σ) suggest good match of Gφ to the perturbed data
distributions. (e-h) Distinguishability, attributability, perturbation length, and orthogonality of 100
StyleGAN user-end models on FFHQ and 100 DCGAN user-end models on MNIST or CelebA,
respectively.

3 EXPERIMENTS AND ANALYSIS
In this section we test Theorem 1, provide empirical support for the orthogonality of keys, and
present experimental results on model attribution using MNIST, CelebA, and FFHQ. Note that tests
on the theorems require estimation of Σ, which is costly for models with high-dimensional outputs,
and therefore are only performed on MNIST and CelebA.

Key generation We generate keys by iteratively solving the following convex problem:

φi = arg min

φ

Ex∼PD,G0

(cid:2)max{1 + φT x, 0}(cid:3) +

i−1
(cid:88)

j=1

max{φT

j φ, 0}.

(9)

The orthogonality penalty is omitted for the ﬁrst key. The solutions are normalized to unit l2 norm
before being inserted into the next problem. We note that PD and PG0 do not perfectly match in
practice, and therefore we draw with equal chance from both distributions during the computation.
G0s are trained using the standard DCGAN architecture for MNIST and CelebA, and StyleGAN for
FFHQ. Training details are deferred to Appendix D.

User-end generative models The training of Gφ follows Alg. 1, where γ is iteratively tuned to
balance generation quality and distinguishability. For each γ, we collect a perturbed dataset Dγ,φ
and solve the following training problem:

E(z,x)∼Dγ,φ

(cid:2)||Gφ(z; θ) − x||2(cid:3) ,

min
θ

(10)

starting from θ = θ0. If the resultant model does not meet the distinguishability requirement due to
the discrepancy between Dγ,φ and Gφ, the perturbation is updated as γ = αγ. In experiments, we
use a standard normal distribution for Pz, and set δ = 10−2 and α = 1.1.

Validation of Theorem 1 Here we validate the sufﬁcient condition for distinguishability. Fig. 2a
compares the LHS and RHS values of Eq. (6) for 100 distinguishable user-end models. The em-
pirical distinguishability of these models are reported in Fig. 2e. Calculation of the RHS of Eq. (6)
requires estimations of µ and Σ. To do this, we sample

(cid:15)(z) = Gφ(z; θ) − G(z; θ0) − γφ

(11)

4

Published as a conference paper at ICLR 2021

using 5000 samples of z ∼ Pz, where Gφ and γ are derived from Alg. 1. Σ and µ are then estimated
for each φ. Fig. 2c and d present histograms of the elements in µ and Σ for two user-end models of
the benchmark datasets. Results in Fig. 2a show that the sufﬁcient condition for distinguishability
(Eq. (6)) is satisﬁed for most of the sampled models through the training speciﬁed in Alg. 1. Lastly,
we notice that the LHS values for MNIST are farther away from the equality line than those for
CelebA. This is because the MNIST data distribution resides at corners of the unit box. Therefore
perturbations of the distribution are more likely to exceed the bounds for pixel values. Clamping
of these invalid pixel values reduces the effective perturbation length. Therefore to achieve distin-
guishability, Alg. 1 seeks γs larger than needed. This issue is less observed in CelebA, where data
points are rarely close to the boundaries. Fig. 2g present the values of γs of all user-end models.

using Dγ,φ ;

Algorithm 1: Training of Gφ
input : φ, G0
output: Gφ, γ
1 set γ = dmax(φ) ;
2 collect Dγ,φ ;
3 train Gφ by solving Eq. (10)

Validation of Theorem 2 Recall that from Theorem 2, we rec-
ognized that orthogonal keys are sufﬁcient. To support this de-
sign rule, Fig. 2b presents the minimum RHS values of Eq. (8)
for 100 user-end models. Speciﬁcally, for each φi, we com-
pute a(φi, φj) (Eq. (7)) using φj for j = 1, ..., i − 1 and re-
port minj a(φi, φj), which sets an upper bound on the angle be-
tween φi and all existing φs. The resultant minj a(φi, φj) are
all positive for MNIST and close to zero for CelebA. From this
result, an angle of ≥ 94 deg, instead of 90 deg, should be en-
forced between any pairs of keys for CelebA. However, since
the conditions are sufﬁcient, orthogonal keys still empirically
achieve high attributability (Fig. 2f), although improvements can
be made by further increasing the angle between keys. Also no-
tice that the current computation of keys (Eq. (9)) does not en-
force a hard constraint on orthogonality, leading to slightly acute
angles (87.7 deg) between keys for CelebA (Fig. 2h). On the
other hand, the positive values in Fig. 2b for MNIST suggests that further reducing the angles be-
tween keys is acceptable if one needs to increase the total capacity of attributable models. However,
doing so would require the derivation of new keys to rely on knowledge about all existing user-end
models (in order to compute Eq. (7)).

4 compute empirical D(Gφ) ;
5 if D(Gφ) < 1 − δ then
set γ = αγ ;
6
goto step 2 ;

7
8 end

Table 1: Empirical average of distinguishability ( ¯D),attributability (A(G)), ||∆x||, and FID scores.
DCGANM (DCGANC) for MNIST (CelebA). Std in parenthesis. FID0: FID for G0. ↓ means lower
is better and ↑ means higher is better.

GANs

Angle

¯D ↑ A(G) ↑ ||∆x|| ↓

DCGANM

DCGANC

StyleGAN

Orthogonal 0.99 0.99
45 degree
0.99 0.13
Orthogonal 0.99 0.93
45 degree
0.99 0.15
Orthogonal 0.99 0.99
0.97 0.18
45 degree

FID ↓

3.97 (0.29)
3.85 (0.12)
4.04 (0.32)
4.57 (0.35)
36.04 (23.35) 12.43(0.16)
67.19 (35.53)

FID0 ↓
10.43 (0.77)
7.82 (0.15)
-
11.02 (0.86)
35.95 (0.12) 58.69 (5.21)
59.81 (5.34)
-
35.23(14.67)
47.86(10.66)

-

Empirical results on benchmark datasets Tab. 1 reports the metrics of interest measured on the
100 user-end models for each of MNIST and CelebA, and 20 models for FFHQ. All models are
trained to be distinguishable. And by utilizing Theorem 2, they also achieve high attributability. As
a comparison, we demonstrate results where keys are 45 deg apart (φT φ(cid:48) = 0.71) using a separate
set of 20 user-end models for each of MNIST and CelebA, and 5 models for FFHQ, in which case
distinguishability no longer guarantees attributability. Regarding generation quality, Gφs receive
worse FID scores than G0 due to the perturbations. We visualize samples from user-end models and
the corresponding keys in Fig. 3. Note that for human faces, FFHQ in particular, the perturbations
create light shades around eyes and lips, which is an unexpected but reasonable result.

Attribution robustness vs. generation quality We now consider the scenario where outputs of
the generative models are post-processed (e.g., by adversaries) before being attributed. When the
post-processes are known, we can take counter measures through robust training, which intuitively

5

Published as a conference paper at ICLR 2021

Figure 3: Visualization of sample keys (1st row) and the corresponding user-end generated contents.

will lead to additional loss of generation quality. To assess this tradeoff between robustness and
generation quality, we train Gφ against post-processes T : Rdx → Rdx from a distribution PT . Due
to the potential nonlinearity of T and the lack of theoretical guarantee in this scenario, we resort to
the following robust training problem for deriving the user-end models:

Ez∼Pz,T ∈PT

(cid:2)max{1 − fφi(T (Gφi(z; θi))), 0} + C||G0(z) − Gφi(z; θi)||2(cid:3) ,

(12)

min
θi

where C is the hyper-parameter for generation quality. Detailed analysis and comparison for se-
lecting C are provided in Appendix E. We consider ﬁve types of post-processes: blurring, crop-
ping, noise, JPEG conversion and the combination of these four. Examples of the post-processed
images are shown in Fig. 5. Blurring uses Gaussian kernel widths uniformly drawn from
1
3 {1, 3, 5, 7, 9}. Cropping crops images with uniformly drawn ratios between 80% and 100%, and
scales the cropped images back to the original size using bilinear interpolation. Noise adds white
noise with standard deviation uniformly drawn from [0, 0.3]. JPEG applies JPEG compression.
Combination performs each attack with a 50% chance in the order of Blurring, Cropping,
Noise and JPEG. For differentiability, we use existing implementations of differentiable blur-
ring (Riba et al. (2020)) and JPEG conversion (Zhu et al. (2018)). For robust training, we apply the
post-process to mini-batches with 50% probability.

We performed comprehensive tests using DCGAN (on MNIST and CelebA), PGAN (on CelebA),
and CycleGAN (on Cityscapes). Tab. 2 summarizes the average distinguishability, the attributabil-
ity, the perturbation length ||∆x||, and the FID score with and without robust training of Gφ. Re-
sults are based on 20 models for each architecture-dataset pair, where keys are kept orthogonal and
data compliant. From the results, defense against these post-processes can be achieved, except for
Combination. Importantly, there is a clear tradeoff between robustness and generation quality.
This can be seen from Fig. 5, which compares samples with7 and without robust training from the
tested models and datasets.

Lastly, it is worth noting that the training formulation in Eq. (12) can also be applied to the
training of non-robust user-end models in place of Eq. (10). However, the resultant model from
Eq. (12) cannot be characterized by Eq. (5) with small µ and Σ, i.e., due to the nonlinearity of
the training process of Eq. (12, the user-end model distribution is deformed while it is perturbed.
This resulted in unsuccessful validation of the theorems, which led to the adoption of Eq. (10) for
theorem-consistent training. Therefore, while the empirical results show feasibility of achieving
robust attributability using Eq. (12, counterparts to Theorems 1 and 2 in this nonlinear setting are
yet to be developed.

6

Published as a conference paper at ICLR 2021

Table 2: DCGANM: MNIST. DCGANC: CelebA. Dis.: Distinguishability before (Bfr) and after
(Aft) robust training. Att.: Attributability. ||∆x|| and FID are after robust training. Std in paren-
thesis. ↓ means lower is better and ↑ means higher is better. ||∆x|| and FID before robust train-
ing: DCGANM:||∆x|| = 5.05, FID = 5.36. DCGANC:||∆x|| = 5.63, FID = 53.91. PGAN:
||∆x|| = 9.29, FID = 21.62. CycleGAN: ||∆x|| = 55.85. FID does not apply to CycleGAN.

Metric Model
-

-

Blurring Cropping
Bfr Aft

Bfr Aft

Noise
Bfr Aft

JPEG
Bfr Aft

Combi.
Bfr Aft

Dis. ↑

Att. ↑

||∆x|| ↓

DCGANM
DCGANC
PGAN
CycleGAN 0.49

0.96
0.49
0.49
0.99
0.50 0.98
0.92

0.94
DCGANM
0.98
DCGANC
PGAN
0.99
CycleGAN 0.08 0.98

0.02
0.00
0.13

15.96(2.18)
DCGANM
11.83(0.65)
DCGANC
PGAN
18.49(2.04)
CycleGAN 68.03(3.62)

FID ↓

DCGANM
DCGANC
PGAN

41.11(20.43)
73.62(6.70)
28.15(3.43)

0.52
0.49
0.51
0.49

0.03
0.00
0.07
0.05

0.99
0.99
0.99
0.87

0.88
0.99
0.99
0.93

0.85
0.95
0.97
0.98

0.77
0.89
0.97
0.97

0.99
0.98
0.99
0.99

0.95
0.93
0.99
0.98

0.54
0.51
0.96
0.55

0.16
0.07
0.99
0.47

0.99
0.99
0.99
0.99

0.98
0.98
0.99
0.99

0.50
0.50
0.50
0.49

0.00
0.00
0.06
0.05

0.66
0.85
0.76
0.67

0.26
0.70
0.98
0.73

9.17(0.65)
9.30(0.31)
21.27(0.81)
80.03(3.59)

21.58(2.44)
98.86(9.51)
47.94(5.71)

5.93(0.34)
4.75(0.17)
10.20(0.81)
55.47(1.60)

5.79(0.19)
59.51(1.60)
25.43(2.19)

6.48(0.94)
6.01(0.29)
10.08(1.03)
57.42(2.00)

6.50(1.70)
60.35(2.57)
22.86(2.06)

17.08(1.86)
13.69(0.59)
24.82(2.33)
83.94(4.66)

68.16(24.67)
87.29(9.29)
45.16(7.87)

4 DISCUSSION

Capacity of keys For real-world applications, we hope to
maintain attributability for a large set of keys. Our study so
far suggests that the capacity of keys is constrained by the data
compliance and orthogonality requirements. While the empir-
ical study showed the feasibility of computing keys through
Eq. (9), ﬁnding the maximum number of feasible keys is a
problem about optimal sphere packing on a segment of the unit
sphere (Fig. 4). To explain, the unit sphere represents the iden-
tiﬁability requirement ||φ|| = 1. The feasible segment of the
unit sphere is determined by the data compliance and gener-
ation quality constraints. And the spheres to be packed have
radii following the sufﬁcient condition in Theorem 2. Such
optimal packing problems are known open challenges (Cohn
et al. (2017); Cohn (2016)). For real-world applications where
a capacity of attributable models is needed (which is the case
for both malicious personation and copyright infringement set-
tings), it is necessary to ﬁnd approximated solutions to this problem.

Figure 4: Capacity of keys as a
sphere packing problem: The fea-
sible space (arc) is determined by
the data compliance and genera-
tion quality constraints, and the
size of spheres by the minimal an-
gle between keys.

Generation quality control From Proposition 1 and Theorem 1, the inevitable loss of generation
quality is directly related to the length of perturbation (γ), which is related to dmax. Fig. 6 com-
pares outputs from user-end models with different dmaxs. While it is possible to ﬁlter φs based on
their corresponding dmaxs for generation quality control, here we discuss a potential direction for
prescribing a subspace of φs within which quality can be controlled. To start, we denote by J(x)
the Jacobian of G0 with respect to its generator parameters θ0. Our discussion is related to the ma-
trix M = Ex∼PG0
[J(x)T ]. A spectral analysis of M reveals that the eigenvectors of
M with large eigenvalues are more structured than those with small ones (Fig. 7(a)). This ﬁnding
is consistent with the deﬁnition of M : The largest eigenvectors of M represent the principal axes
of all mean sensitivity vectors, where the mean is taken over the latent space. For MNIST, these
eigenvectors overlap with the digits; for CelebA, they are structured color patterns. On the other
hand, the smallest eigenvectors represent directions rarely covered by the sensitivity vectors, thus

[J(x)]Ex∼PG0

7

Published as a conference paper at ICLR 2021

Figure 5: Samples from user-end models with robust and non-robust training. For each subﬁgure -
top: DCGAN on MNIST and CelebA; bottom: PGAN (CelebA) and CycleGAN (Cityscapes). For
each dataset - top: samples from G0 (after worst-case post-process in (b-f)); mid: samples from
Gφ (after robust training in (b-f)); btm (a): difference between non-robust Gφ and G0; btm (b-h)
difference between robust and non-robust Gφ.

Figure 6: MNIST, CelebA, and FFHQ examples from Gφs with (a-c) small dmax and (d-f) large
dmax. All models are distinguishable and attributable. (Zooming in on pdf ﬁle is recommended.)

resembling random noise. Based on this ﬁnding, we test the hypothesis that keys more aligned with
the eigenspace of the small eigenvalues will have smaller dmax. We test this hypothesis by com-
puting the Pearson correlations between dmax and φT M φ using 100 models for each of MNIST
and CelebA. The resultant correlations are 0.33 and 0.53, respectively. In addition, we compare
outputs from models using the largest and the smallest eigenvectors of M as the keys in Fig. 7b.
While a concrete human study is needed, the visual results suggest that using eigenvectors of M is a
promising approach to segmenting the space of keys according to their induced generation quality.

5 RELATED WORK

Detection and attribution of model-generated contents This paper focused on the attribution of
contents from generative models rather than the detection of hand-crafted manipulations (Agarwal &
Farid (2017); Popescu & Farid (2005); O’brien & Farid (2012); Rao & Ni (2016); Huh et al. (2018)).
Detection methods rely on ﬁngerprints intrinsic to generative models (Odena et al. (2016); Zhang

8

Published as a conference paper at ICLR 2021

Figure 7: (a) Eigenvectors for the two largest and two smallest eigenvalues of M for DCGANs on
MNIST (top) and CelebA (bottom). (b) Left column: Samples from G0; Rest: G0 − Gφ where φ
are the eigenvectors in (a).

et al. (2019b); Marra et al. (2019); Wang et al. (2019)), yet similar ﬁngerprints present for models
trained on similar datasets (Marra et al. (2019)). Thus ﬁngerprints cannot be used for attribution.
Skripniuk et al. (2020) studied decentralized attribution. Instead of using linear classiﬁers for attri-
bution, they train a watermark encoder-decoder network that embeds (and reads) watermarks into
(and from) the content, and compare the decoded watermark with user-speciﬁc ones. Their method
does not provide sufﬁcient conditions of the watermarks for attributability.

IP protection of digital contents and models Watermarks have conventionally been used for IP
protection (Tirkel et al., 1993; Van Schyndel et al., 1994; Bi et al., 2007; Hsieh et al., 2001; Pereira
& Pun, 2000; Zhu et al., 2018; Zhang et al., 2019a) without considering the attribution guarantee.
Another approach to content IP protection is blockchain (Hasan & Salah, 2019). However, this
approach requires meta data to be transferred along with the contents, which may not be realistic in
adversarial settings. E.g., one can simply take a picture of a synthetic image to remove any meta
data attached to the image ﬁle. Aside from the protection of contents, mechanisms for protecting
IP of models have also been studied (Uchida et al., 2017; Nagai et al., 2018; Le Merrer et al., 2019;
Adi et al., 2018; Zhang et al., 2018; Fan et al., 2019; Szyller et al., 2019; Zhang et al., 2020).
Model watermarking is usually done by adding watermarks into model weights (Uchida et al., 2017;
Nagai et al., 2018), by embedding unique input-output mapping into the model (Le Merrer et al.,
2019; Adi et al., 2018; Zhang et al., 2018), or by introducing a passport mechanism so that model
accuracy drops if the right passport is not inserted (Fan et al., 2019). While closely related, existing
work on model IP protection focused on the distinguishability of individual models, rather than the
attributability of a model set.

6 CONCLUSION

Motivated by emerging challenges with generative models, e.g., deepfake, this paper investigated
the feasibility of decentralized attribution of such models. The study is based on a protocol where
the registry generates user-speciﬁc keys that guides the watermarking of user-end models to be
distinguishable from the authentic data. The outputs of user-end models will then be attributed
by the registry through the binary classiﬁers parameterized by the keys. We developed sufﬁcient
conditions of the keys so that distinguishable user-end models achieve guaranteed attributability.
These conditions led to simple rules for designing the keys. With concerns about adversarial post-
processes, we further showed that robust attribution can be achieved using the same design rules,
and with additional loss of generation quality. Lastly, we introduced two open challenges towards
real-world applications of the proposed attribution scheme: the prescription of the key space with
controlled generation quality, and the approximation of the capacity of keys.

7 ACKNOWLEDGEMENTS

Support from NSF Robust Intelligence Program (1750082), ONR (N00014-18-1-2761), and Ama-
zon AWS MLRA is gratefully acknowledged. We would like to express our gratitude to Ni Trieu
(ASU) for providing us invaluable advice, and Zhe Wang, Joshua Feinglass, Sheng Cheng, Yong-
baek Cho and Huiliang Shao for helpful comments.

9

Published as a conference paper at ICLR 2021

REFERENCES

Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, and Joseph Keshet. Turning your weak-
ness into a strength: Watermarking deep neural networks by backdooring. In 27th {USENIX}
Security Symposium ({USENIX} Security 18), pp. 1615–1631, 2018.

Shruti Agarwal and Hany Farid. Photo forensics from jpeg dimples. In 2017 IEEE Workshop on

Information Forensics and Security (WIFS), pp. 1–6. IEEE, 2017.

Ning Bi, Qiyu Sun, Daren Huang, Zhihua Yang, and Jiwu Huang. Robust image watermarking
based on multiband wavelets and empirical mode decomposition. IEEE Transactions on Image
Processing, 16(8):1956–1966, 2007.

Ali Breland. The bizarre and terrifying case of the “deepfake” video that helped bring an african
nation to the brink. motherjones, Mar 2019. URL https://www.motherjones.com/
politics/2019/03/deepfake-gabon-ali-bongo/.

Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high ﬁdelity natural

image synthesis. arXiv preprint arXiv:1809.11096, 2018.

Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. Star-
gan: Uniﬁed generative adversarial networks for multi-domain image-to-image translation.
In
Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8789–8797,
2018.

Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis

for multiple domains. arXiv preprint arXiv:1912.01865, 2019.

Henry Cohn. A conceptual breakthrough in sphere packing. arXiv preprint arXiv:1611.01685, 2016.

Henry Cohn, Abhinav Kumar, Stephen D Miller, Danylo Radchenko, and Maryna Viazovska. The

sphere packing problem in dimension 24. Annals of Mathematics, pp. 1017–1033, 2017.

Lixin Fan, Kam Woh Ng, and Chee Seng Chan. Rethinking deep neural network ownership veri-
ﬁcation: Embedding passports to defeat ambiguity attacks. In Advances in Neural Information
Processing Systems, pp. 4716–4725, 2019.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural infor-
mation processing systems, pp. 2672–2680, 2014.

Haya R Hasan and Khaled Salah. Combating deepfake videos using blockchain and smart contracts.

Ieee Access, 7:41596–41606, 2019.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.
Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances
in Neural Information Processing Systems, pp. 6626–6637, 2017.

Ming-Shing Hsieh, Din-Chang Tseng, and Yong-Huai Huang. Hiding digital watermarks using
multiresolution wavelet transform. IEEE Transactions on industrial electronics, 48(5):875–882,
2001.

Minyoung Huh, Andrew Liu, Andrew Owens, and Alexei A Efros. Fighting fake news: Image splice
detection via learned self-consistency. In Proceedings of the European Conference on Computer
Vision (ECCV), pp. 101–117, 2018.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for im-

proved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative
adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pp. 4401–4410, 2019a.

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyz-

ing and improving the image quality of stylegan. arXiv preprint arXiv:1912.04958, 2019b.

10

Published as a conference paper at ICLR 2021

Kelly.

Makena
fakes.
2019.
deep-fakes-regulation-facebook-adam-schiff-congress-artificial-intelligence.

deep-
grapples
Jun
https://www.theverge.com/2019/6/13/18677847/

regulate
deepfakes,

Congress
URL

to
regulate

grapples
with

Congress

with

how

how

to

Erwan Le Merrer, Patrick Perez, and Gilles Tr´edan. Adversarial frontier stitching for remote neural

network watermarking. Neural Computing and Applications, pp. 1–12, 2019.

Yann LeCun and Corinna Cortes. MNIST handwritten digit database. 2010. URL http://yann.

lecun.com/exdb/mnist/.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild.
In Proceedings of the IEEE international conference on computer vision, pp. 3730–3738, 2015.

Francesco Marra, Diego Gragnaniello, Luisa Verdoliva, and Giovanni Poggi. Do gans leave artiﬁcial
In 2019 IEEE Conference on Multimedia Information Processing and Retrieval

ﬁngerprints?
(MIPR), pp. 506–511. IEEE, 2019.

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization

for generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.

Yuki Nagai, Yusuke Uchida, Shigeyuki Sakazawa, and Shin’ichi Satoh. Digital watermarking for
International Journal of Multimedia Information Retrieval, 7(1):3–16,

deep neural networks.
2018.

James F O’brien and Hany Farid. Exposing photo manipulation with inconsistent reﬂections. ACM

Trans. Graph., 31(1):4–1, 2012.

Augustus Odena, Vincent Dumoulin, and Chris Olah. Deconvolution and checkerboard arti-
facts. Distill, 2016. doi: 10.23915/distill.00003. URL http://distill.pub/2016/
deconv-checkerboard.

Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context
encoders: Feature learning by inpainting. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pp. 2536–2544, 2016.

Shelby Pereira and Thierry Pun. Robust template matching for afﬁne resistant image watermarks.

IEEE transactions on image Processing, 9(6):1123–1129, 2000.

Alin C Popescu and Hany Farid. Exposing digital forgeries by detecting traces of resampling. IEEE

Transactions on signal processing, 53(2):758–767, 2005.

Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with
deep convolutional generative adversarial networks, 2015. URL http://arxiv.org/abs/
1511.06434. cite arxiv:1511.06434Comment: Under review as a conference paper at ICLR
2016.

Yuan Rao and Jiangqun Ni. A deep learning approach to detection of splicing and copy-move
forgeries in images. In 2016 IEEE International Workshop on Information Forensics and Security
(WIFS), pp. 1–6. IEEE, 2016.

E. Riba, D. Mishkin, D. Ponsa, E. Rublee, and G. Bradski. Kornia: an open source differen-
tiable computer vision library for pytorch, 2020. URL https://arxiv.org/pdf/1910.
02190.pdf.

Raphael Satter. Experts: Spy used ai-generated face to connect with targets. Experts: Spy
used AI-generated face to connect with targets, Jun 2019. URL https://apnews.com/
bc2f19097a4c4fffaa00de6770b8a60d.

Vladislav Skripniuk, Ning Yu, Sahar Abdelnabi, and Mario Fritz. Black-box watermarking for

generative adversarial networks. arXiv preprint arXiv:2007.08457, 2020.

Sebastian Szyller, Buse Gul Atli, Samuel Marchal, and N Asokan. Dawn: Dynamic adversarial

watermarking of neural networks. arXiv preprint arXiv:1906.00830, 2019.

11

Published as a conference paper at ICLR 2021

Anatol Z Tirkel, GA Rankin, RM Van Schyndel, WJ Ho, NRA Mee, and Charles F Osborne. Elec-
tronic watermark. Digital Image Computing, Technology and Applications (DICTA’93), pp. 666–
673, 1993.

Yusuke Uchida, Yuki Nagai, Shigeyuki Sakazawa, and Shin’ichi Satoh. Embedding watermarks
In Proceedings of the 2017 ACM on International Conference on

into deep neural networks.
Multimedia Retrieval, pp. 269–277, 2017.

Ron G Van Schyndel, Andrew Z Tirkel, and Charles F Osborne. A digital watermark. In Proceedings

of 1st International Conference on Image Processing, volume 2, pp. 86–90. IEEE, 1994.

Sheng-Yu Wang, Oliver Wang, Andrew Owens, Richard Zhang, and Alexei A Efros. Detecting
photoshopped faces by scripting photoshop. In Proceedings of the IEEE International Conference
on Computer Vision, pp. 10072–10081, 2019.

Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. High-
resolution image synthesis and semantic manipulation with conditional gans. In Proceedings of
the IEEE conference on computer vision and pattern recognition, pp. 8798–8807, 2018.

Ning Yu, Larry Davis, and Mario Fritz. Attributing fake images to gans: Analyzing ﬁngerprints in

generated images. arXiv preprint arXiv:1811.08180, 2018.

Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dim-
itris N. Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adver-
sarial networks. In The IEEE International Conference on Computer Vision (ICCV), Oct 2017.

Jialong Zhang, Zhongshu Gu, Jiyong Jang, Hui Wu, Marc Ph Stoecklin, Heqing Huang, and Ian
Molloy. Protecting intellectual property of deep neural networks with watermarking. In Proceed-
ings of the 2018 on Asia Conference on Computer and Communications Security, pp. 159–172,
2018.

Jie Zhang, Dongdong Chen, Jing Liao, Han Fang, Weiming Zhang, Wenbo Zhou, Hao Cui, and
Nenghai Yu. Model watermarking for image processing networks. In AAAI, pp. 12805–12812,
2020.

Kevin Alex Zhang, Alfredo Cuesta-Infante, Lei Xu, and Kalyan Veeramachaneni. Steganogan: High

capacity image steganography with gans. arXiv preprint arXiv:1901.03892, 2019a.

Xu Zhang, Svebor Karaman, and Shih-Fu Chang. Detecting and simulating artifacts in gan fake

images. arXiv preprint arXiv:1907.06515, 2019b.

Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. Hidden: Hiding data with deep networks.
In Proceedings of the European Conference on Computer Vision (ECCV), pp. 657–672, 2018.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation
using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference
on computer vision, pp. 2223–2232, 2017.

12

Published as a conference paper at ICLR 2021

A PROOF OF PROPOSITION 1

Proposition 1. Let dmax(φ) := maxx∼PD |φT x|. If ε ≥ 1 + dmax(φ), then ∆x = (1 + dmax(φ))φ
solves Eq. (4), and fφ(x + ∆x) > 0 ∀ x ∼ G0.

Proof. Let φ be a data-compliant key and let x be sampled from PD. First, from the KKT conditions
for Eq. (4) we can show that the solution ∆x∗ is proportional to φ:

∆x∗ = φ/µ∗,

where µ∗ ≥ 0 is the Lagrange multiplier. To minimize the objective, we seek µ such that

1 − (x + ∆x∗)T φ = 1 − xT φ − 1/µ∗ ≤ 0,

(13)

(14)

for all x. Since xT φ < 0 (data compliance), this requires 1/µ∗ = 1 + dmax(φ). Therefore, when
ε ≥ 1 + dmax(φ), ∆x∗ = (1 + dmax(φ))φ solves Eq. (4). And fφ(x + ∆x∗) = φT (x + (1 +
dmax(φ))φ) = φT x + 1 + dmax(φ) > 0.

B PROOF OF THEOREM 1

Theorem 1. Let dmax(φ) = maxx∈D |φT x|, σ2(φ) = φT Σφ, δ ∈ [0, 1], and φ be a data-compliant
key. D(Gφ) ≥ 1 − δ/2 if

(cid:115)

γ ≥ σ(φ)

log

(cid:19)

(cid:18) 1
δ2

+ dmax(φ) − φT µ.

(15)

(cid:2)1(φT x < 0)(cid:3) = 1. Therefore
Proof. We ﬁrst note that due to data compliance of keys, Ex∼PD
(cid:2)1(φT x > 0)(cid:3) ≥ 1 − δ, i.e., Pr(φT x > 0) ≥ 1 − δd for x ∼ PGφ.
D(Gφ) ≥ 1 − δ/2 iff Ex∼PGφ
We now seek a lower bound for Pr(φT x > 0). To do so, let x and x0 be sampled from PGφ and
PG0, respectively. Then we have

φT x = φT (x0 + γφ + (cid:15))
= φT x0 + γ + φT (cid:15),

and

Pr(φT x > 0) = Pr (cid:0)φT (cid:15) > −φT x0 − γ(cid:1) .

Since dmax(φ) ≥ −φT x0, we have

(16)

(17)

Pr(φT x > 0) ≥ Pr (cid:0)φT (cid:15) > dmax(φ) − γ(cid:1) = Pr (cid:0)φT ((cid:15) − µ) ≤ γ − dmax(φ) + φT µ(cid:1) .

(18)

The latter sign switching in equation 18 is granted by the symmetry of the distribution of φT ((cid:15) − µ),
which follows N (0, φT Σφ). A sufﬁcient condition for Pr(φT x > 0) ≥ 1 − δ is then

Pr (cid:0)φT ((cid:15) − µ) ≤ γ − dmax(φ) + φT µ(cid:1) ≥ 1 − δ.

Recall the following tail bound of x ∼ N (0, σ2) for y ≥ 0:

Pr(x ≤ σy) ≥ 1 − exp(−y2/2).

Compare equation 20 with equation 19, the sufﬁcient condition becomes

(cid:115)

γ ≥ σ(φ)

log

(cid:19)

(cid:18) 1
δ2

+ dmax(φ) − φT µ.

(19)

(20)

(21)

13

Published as a conference paper at ICLR 2021

C PROOF OF THEOREM 2

Theorem 2. Let dmin = minx∈D |φT x|, dmax = maxx∈D |φT x|, σ2(φ) = ||φ||2
1 − N δ if D(G) ≥ 1 − δ for all Gφ ∈ G and for any pair of data-compliant keys φ and φ(cid:48):

Σ, δ ∈ [0, 1]. A(G) ≥

φT φ(cid:48) ≤ −1 +

dmax(φ(cid:48)) + dmin(φ(cid:48)) − 2φ(cid:48)T µ

(cid:113)

σ(φ(cid:48))

log (cid:0) 1
δ2

(cid:1) + dmax(φ(cid:48)) − φ(cid:48)T µ

.

(22)

Proof. Let φ and φ(cid:48) be any pair of keys. Let x and x0 be sampled from PGφ and PG0, respectively.
We ﬁrst derive the sufﬁcient conditions for Pr(φ(cid:48)T x < 0) ≥ 1 − δ. Since x = x0 + γφ + (cid:15) for
x ∈ Gφ, we have

φ(cid:48)T x = φ(cid:48)T (x0 + γφ + (cid:15))

= φ(cid:48)T x0 + γφT φ(cid:48) + φ(cid:48)T (cid:15).

Then

Pr(φ(cid:48)T x < 0) = Pr (cid:0)φ(cid:48)T (cid:15) < −φ(cid:48)T x0 − γφT φ(cid:48)(cid:1)

≥ Pr (cid:0)φ(cid:48)T ((cid:15) − µ) < dmin(φ(cid:48)) − γφT φ(cid:48) − φ(cid:48)T µ(cid:1) ,

(23)

(24)

where dmin(φ(cid:48)) := minx∈D |φ(cid:48)T x| and φ(cid:48)T ((cid:15) − µ) ∼ N (0, σ2(φ(cid:48))). Using the same tail bound of
normal distribution and Theorem 1, we have Pr(φT x < 0) ≥ 1 − δ if

(cid:115)

− γφT φ(cid:48) ≥ σ(φ(cid:48))

log

(cid:19)

(cid:18) 1
δ2

− dmin(φ(cid:48)) + φ(cid:48)T µ

⇒ φT φ(cid:48) ≤ −1 +

dmax(φ(cid:48)) + dmin(φ(cid:48)) − 2φ(cid:48)T µ

(cid:113)

σ(φ(cid:48))

log (cid:0) 1
δ2

(cid:1) + dmax(φ(cid:48)) − φ(cid:48)T µ

(25)

Note that Pr(A = 1, B = 1) = 1 − Pr(A = 0) − Pr(B = 0) + Pr(A = 0, B = 0) ≥ 1 − Pr(A =
0) − Pr(B = 0) for binary random variables A and B. With this, it is straight forward to show
(cid:54)= φ, and Pr(φT x > 0) ≥ 1 − δ for all φ, then
that when Pr(φ(cid:48)T x < 0) ≥ 1 − δ for all φ(cid:48)
Pr(φT x > 0, φ(cid:48)T x < 0 ∀φ(cid:48) (cid:54)= φ) ≥ 1 − N δ and A(G) ≥ 1 − N δ.

D TRAINING DETAILS

D.1 METHOD

We trained user-end models based on the objective function (Eq.(10) in the main text). For datasets
where the root models follow DCGAN and PGAN, the user-end models follow the same architec-
ture. For the FFHQ dataset where StyleGAN is used, we introduce an additional shallow convolu-
tional network as a residual part, which is added to the original StyleGAN output to match with the
perturbed datasets Dγ,φ. In this case, the training using Eq.(10) is limited to the additional shallow
network, while the StyleGAN weights are frozen. More speciﬁcally, denoting the combination of
convolution, ReLU, and max-pooling by Conv-ReLU-Max, the shallow network consists of three
Conv-ReLU-Max blocks and one fully connected layer. All of the convolution layers have 4 x 4
kernels, stride 2, and padding 1. And all of the max-pooling layers have 3 x 3 kernels and stride 2.

D.2 PARAMETERS

We adopt the Adam optimizer for training. Training hyper-parameters are summarized in Table 3.

14

Published as a conference paper at ICLR 2021

Table 3: Hyper-parameters to train keys (φ) and generators (Gφ).

GANs

Dataset Batch Size Learning Rate

DCGAN
DCGAN
StyleGAN FFHQ

MNIST 16
CelebA 64
8

0.001
0.001
0.001

β1
0.9
0.9
0.9

β2
0.99
0.99
0.99

Epochs

10
2
5

D.3 TRAINING TIME

All experiments are conducted on V100 Tesla GPUs. Table 4 summarizes the number of GPUs
used and the training time for the non-robust models (Eq.(10) in the main text) and robust models
(Eq.(12) in the main text). Recall that we chose Eq.(10) for training the non-robust user-end models
for consistency with the theorems, although Eq.(12) can be used to achieve attributability in practice,
as is shown in the robust attribution study. Therefore, the non-robust training takes longer to due the
iteration of γ in Alg. 1.

Table 4: Training time (in minute) of one key (Eq.(9) in main text) and one generator (Eq.(10) in
main text). DCGANM: DCGAN for MNIST, DCGANC: DCGAN for CelebA.

GANs

GPUs Key Non-robust Blurring Cropping Noise

JPEG Combination

DCGANM 1
1
DCGANC
PGAN
2
CycleGAN 1
StyleGAN 1

14
15

1.77
5.31
50.89 141.07
20.88 16.04
54.23 3.12

4.12
10.33
140.05
16.26
-

3.96
9.56
131.90
15.43
-

5.12
10.76

5.71
4.19
10.35
10.25
133.46 132.46 135.07
16.41
15.98
15.71
-
-
-

E ABLATION STUDY

Here we conduct an ablation study on the hyper-parameter C for the robust training formulation
(Eq.(12)). Training with larger C focuses more on generation quality, thus sacriﬁcing distinguisha-
bility and attributability. These effects are reported in Table 5 and Table 6. Due to limited time, the
results here are averaged over ﬁve models for each C and data-model pairs.

15

Published as a conference paper at ICLR 2021

Table 5: Distinguishability (top), attributability (btm) before (Bfr) and after (Aft) robust training.
DCGANM: DCGAN for MNIST, DCGANC: DCGAN for CelebA.

Model
-

C
-

Blurring Cropping
Bfr Aft

Bfr Aft

Noise
Bfr Aft

JPEG
Bfr Aft

Combination
Bfr Aft

0.97
0.49
10
DCGANM
0.61
0.49
100
DCGANM
0.50
0.49
1K
DCGANM
0.99
0.49
10
DCGANC
0.96
0.50
100
DCGANC
0.62
0.50
1K
DCGANC
0.98
0.50
100
PGAN
1K
0.89
0.50
PGAN
10K 0.50 0.61
PGAN
0.92
CycleGAN 1K
0.49
0.70
CycleGAN 10K 0.49

0.94
DCGANM
0.02
10
DCGANM
0.87
0.00
100
DCGANM
0.75
0.00
1K
0.98
DCGANC
0.00
10
DCGANC
0.95
0.00
100
DCGANC
0.90
0.00
1K
1.00
PGAN
0.26
100
PGAN
1K
0.99
0.21
10K 0.00 0.51
PGAN
0.99
0.00
CycleGAN 1K
0.87
CycleGAN 10K 0.00

0.51 0.99
0.51
0.98
0.81
0.51
0.49 0.99
0.99
0.49
0.49
0.97
0.50 0.99
0.95
0.49
0.50
0.76
0.50 0.87
0.66
0.50

0.03 0.88
0.85
0.00
0.00
0.80
0.00 0.99
0.93
0.00
0.00
0.89
0.21 1.00
0.99
0.00
0.00
0.90
0.00 0.97
0.77
0.00

0.99
0.97

0.53
0.53

0.99 0.53 0.99 0.50
0.84
0.98
0.50
0.76
0.50
0.91
0.69
0.99 0.50 0.99 0.49
0.96
0.49
0.93
0.92
0.99
0.50
0.50
0.88
0.49
0.91
0.99
0.96 0.99 0.50
0.96 0.99
0.50
0.99
0.88
0.95
0.94
0.76
0.89
0.50
0.98
0.90
0.55 0.99 0.49
0.98 0.99
0.50
0.98
0.52
0.96
0.94

0.10
0.10

0.95
0.91

0.95 0.16 0.98 0.00
0.77
0.00
0.90
0.73
0.80
0.00
0.63
0.93 0.07 0.98 0.00
0.89
0.00
0.85
0.82
0.93
0.02
0.00
0.77
0.00
0.81
0.88
0.99 0.99 0.00
0.99 0.99
0.00
0.99
0.98
0.98
0.97
0.83
0.90
0.00
0.99
0.92
0.45 0.99 0.00
0.97 0.99
0.00
0.99
0.30
0.96
0.95

0.63
0.52
0.51
0.85
0.61
0.51
0.81
0.60
0.51
0.62
0.51

0.26
0.13
0.05
0.70
0.61
0.43
0.99
0.54
0.22
0.77
0.31

Table 6: ||∆x|| (top) and FID score (btm). Standard deviations in parenthesis. DCGANM: DCGAN
for MNIST, DCGANC: DCGAN for CelebA, Combi.: Combination attack. Lower is better.

Model

C

Baseline

Blurring Cropping Noise

JPEG

Combi.

DCGANM 10
5.05(0.09)
15.96(2.18)
DCGANM 100 4.09(0.53)
12.95(4.47)
DCGANM 1K 3.88(0.60)
7.17(2.10)
5.63(0.11)
10
DCGANC
11.83(0.65)
100 3.08(0.27)
DCGANC
10.00(1.61)
1K 2.55(0.36)
7.68(1.53)
DCGANC
100 9.29(0.95)
PGAN
18.49(2.04)
PGAN
1K 6.52(1.85)
14.79(4.15)
10.19(2.87)
10K 5.04(1.63)
PGAN
CycleGAN 1K 55.85(3.67) 68.03(3.62)
CycleGAN 10K 49.66(5.01) 58.64(3.70)

9.17(0.65)
7.62(1.55)
7.43(1.37)
9.30(0.31)
7.80(0.58)
7.13(0.47)
21.27(0.81)
18.88(1.96)
18.23(0.94)
80.03(3.59)
66.05(3.47)

17.08(1.86)
6.48(0.94)
5.93(0.34)
12.70(3.37)
4.70(1.02)
4.57(0.78)
7.56(1.41)
5.12(1.94)
4.22(0.77)
13.69(0.59)
6.01(0.29)
4.75(0.17)
11.65(1.48)
4.26(0.59)
3.20(0.45)
2.65(0.24)
9.23(1.22)
3.39(0.58)
10.20(0.81) 10.08(1.03) 24.82(2.33)
6.40(1.48)
22.09(2.12)
7.09(1.62)
17.26(1.39)
5.67(1.62)
5.13(1.14)
55.47(1.60) 57.42(2.00) 83.94(4.66)
53.14(0.44) 54.52(2.30) 66.24(5.29)

41.11(20.43) 21.58(2.44)
23.83(14.29) 18.39(3.70)
18.08(1.77)
10.85(4.28)
98.86(9.51)
53.91(2.20) 73.62(6.70)

68.16(24.67)
6.50(1.70)
5.79(0.19)
DCGANM 10
5.36(0.12)
36.05(16.20)
5.46(0.11)
5.41(0.18)
DCGANM 100 5.32(0.11)
5.37(0.14)
21.86(4.16)
5.30(0.96)
DCGANM 1K 5.23(0.12)
10
DCGANC
59.51(1.60) 60.35(2.57) 87.29(9.29)
100 45.02(3.37) 73.12(11.03) 85.50(12.25) 47.60(2.57) 50.48(4.58) 78.11(12.95)
DCGANC
72.11(13.81) 40.87(3.03) 45.46(5.03) 57.13(7.20)
1K 40.85(3.41) 55.63(7.97)
DCGANC
25.43(2.19) 22.86(2.06) 45.16(7.87)
47.94(5.71)
100 21.62(1.73) 28.15(3.43)
PGAN
43.48(12.24) 19.20(2.96) 19.05(2.82) 35.07(8.72)
1K 19.05(3.14) 25.19(5.26)
PGAN
16.94(1.89) 17.39(2.33) 26.63(4.44)
37.01(8.74)
10K 16.75(1.87) 18.96(2.65)
PGAN

16

