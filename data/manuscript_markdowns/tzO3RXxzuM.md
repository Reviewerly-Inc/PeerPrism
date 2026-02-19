Under review as a conference paper at ICLR 2022

STABILITY BASED GENERALIZATION BOUNDS
FOR EXPONENTIAL FAMILY LANGEVIN DYNAMICS

Anonymous authors
Paper under double-blind review

ABSTRACT

We study the generalization of noisy stochastic mini-batch based iterative algo-
rithms based on the notion of stability. Recent years have seen key advances in
data-dependent generalization bounds for noisy iterative learning algorithms such
as stochastic gradient Langevin dynamics (SGLD) based on (Mou et al., 2018; Li
et al., 2020) and related approaches (Negrea et al., 2019; Haghifam et al., 2020).
In this paper, we unify and substantially generalize stability based generalization
bounds and make three technical advances. First, we bound the generalization
error of general noisy stochastic iterative algorithms (not necessarily gradient de-
scent) in terms of expected stability, which in turn can be bounded by the expected
Le Cam Style Divergence (LSD). Such bounds have a O(1/n) sample dependence
√
unlike many existing bounds with O(1/
n) dependence. Second, we introduce
Exponential Family Langevin Dynamics (EFLD) which is a substantial general-
ization of SGLD and which allows exponential family noise to be used with gradi-
ent descent. We establish data-dependent expected stability based generalization
bounds for general EFLD. Third, we consider an important new special case of
EFLD: noisy sign-SGD, which extends sign-SGD by using Bernoulli noise over
{−1, +1}, and we establish optimization guarantees for the algorithm. Further,
we present empirical results on benchmark datasets to illustrate the our bounds
are non-vacuous and quantitatively much sharper than existing bounds.

1

INTRODUCTION

Recent years have seen renewed interest and advances in characterizing generalization performance
of learning algorithms in terms of stability, which considers change in performance of a learning
algorithm based on change of a single training point (Hardt et al., 2016; Bousquet & Elisseeff, 2002;
Li et al., 2020; Mou et al., 2018). For stochastic gradient descent (SGD), Hardt et al. (2016) estab-
lished generalization bounds based on uniform stability (Hardt et al., 2016; Bousquet & Elisseeff,
2002), although the analysis needed rather small step sizes ηt = 1/t which is not useful in practice.
While improving the analysis for SGD has remained a challenge, advances have been made on noisy
SGD algorithms, especially stochastic gradient Langevin dynamics (SGLD) (Welling & Teh, 2011;
Mou et al., 2018; Li et al., 2020), which adds Gaussian noise to the stochastic gradients of marginal
variance σ2
t . In parallel, there has been key developments on related information-theoretic general-
ization bounds applicable to SGLD type algorithms (Negrea et al., 2019; Haghifam et al., 2020; Xu
& Raginsky, 2017; Russo & Zou, 2016; Pensia et al., 2018).

While these developments have led to major advances in analyzing generalization of noisy SGD
algorithms, they each have certain limitations which leave room for further improvements. Using
uniform stability, Mou et al. (2018) established a bound for SGLD of the form K
t which
n
depends on K, the global Lipschitz constant for the loss, and with step size ηt ≤ σt ln 2/K. The
bound has a desirable dependency of O(1/n) on the samples, but has an undesirable dependence on
K, and the step sizes, bounded by σt/K, are too small. Mou et al. (2018) also presented another
n) sample dependence.
bound which addresses some of these issues, but gets an undesirable O(1/
By building on the developments of Russo & Zou (2016); Xu & Raginsky (2017); Pensia et al.
(2018), Negrea et al. (2019) made advances from the information theoretic perspective and estab-
lished bounds for SGLD which have the desirable dependence on the norm of gradient incoherence,
i.e., difference in gradients over different mini-batches, avoids dependence on Lipschitz constant
n) sample
K, and is applicable to unbounded sub-Gaussian losses, but have an undesirable O(1/

t /σ2

(cid:112)(cid:80)

t η2

√

√

1

Under review as a conference paper at ICLR 2022

dependence. Haghifam et al. (2020) made further advances on the problem from an information the-
oretic perspective based on the conditional mutual information framework of Steinke & Zakynthinou
(2020) and obtained generalization bounds based on gradient incoherence with O(1/n) sample de-
pendence, but their analysis holds for full batch Langevin dynamics, not mini-batch SGLD. Li et al.
(2020) made advances on such bounds based on the notion of Bayes-stability, by combining ideas
from PAC-Bayes bounds into stability, and established a bound of the form c
t for
n
bounded losses, where ge(t) is the expected gradient norm square at step t. While the bound avoids
dependency on the Lipschitz constant K, the dependence on the gradient norm makes such bounds
much weaker than the information theoretic bounds of Negrea et al. (2019); Haghifam et al. (2020)
which depend on the norm of gradient incoherence, which are typically orders of magnitude smaller.
Further, the analysis of Li et al. (2020) still needs small step sizes, bounded by σt/K.

t ge(t)/σ2

(cid:112)(cid:80)

t η2

In this paper, we build on the core strengths of such existing approaches, most notably the O(1/n)
sample dependence of stability based bounds (Mou et al., 2018; Li et al., 2020) and the depen-
dence on gradient incoherence for information theoretic bounds (Negrea et al., 2019; Haghifam
et al., 2020), and develop a framework (Section 2) for developing generalization bounds for noisy
stochastic iterative (NSI) algorithms. Our framework considers generalization based on the concept
of expected stability, rather than uniform stability (Hardt et al., 2016; Bousquet & Elisseeff, 2002;
Bousquet et al., 2020; Mou et al., 2018), which yields distribution dependent generalization bounds
and avoids the worst-case setting of uniform stability. Building on Li et al. (2020), we show that
expected stability of general NSI algorithms can be bounded by the expected Le Cam Style Diver-
gence with dependence on parameter distributions from mini-batches differing by one sample. In
Section 3, we introduce Exponential Family Langevin Dynamics (EFLD), a family of noisy gradient
descent algorithms based on exponential family noise. Special cases of EFLD include SGLD and
noisy versions of Sign-SGD or quantized SGD algorithms widely used in practice (Bernstein et al.,
2018a;b; Jin et al., 2020; Alistarh et al., 2017)Our main result provides an expected stability based
generalization bound applicable to any EFLD algorithm with a O(1/n) sample dependence and a
dependence on gradient incoherence, rather than gradient norms. Existing generalization bounds for
SGLD (Li et al., 2020; Negrea et al., 2019) usually use properties of the Gaussian distribution, and
our analysis on EFLD illustrates that this was unnecessary. We also consider optimization guaran-
tees for EFLD and establish such results for noisy Sign-SGD and SGLD. Through experiments on
benchmark datasets (Section 4), we illustrate that our bounds are non-vacuous and quantitatively
much sharper than existing bounds (Li et al., 2020; Negrea et al., 2019).

Related work. Uniform stability has been a classical approach for bounding generalization error
(Bousquet & Elisseeff, 2002; Bousquet et al., 2020; Feldman & Vondrak, 2018; 2019), pioneered by
Rogers & Wagner (1978); Devroye & Wagner (1979). Beyond the aforementioned work, there has
been recent work on differential privacy that analyzes the uniform stability of differentially private
SGD (DP-SGD) (Hardt et al., 2016; Bassily et al., 2020). Beyond uniform stability, information-
theoretic approaches (Russo & Zou, 2016; Xu & Raginsky, 2017) that bounds the generalization
error by the mutual information between the algorithm input S and the algorithm output w, have
been used for deriving generalization bounds for noisy iterative algorithms (Pensia et al., 2018; Bu
et al., 2019). Along this line of literature, Negrea et al. (2019); Haghifam et al. (2020); Rodr´ıguez-
G´alvez et al. (2021) prove data-dependent generalization bounds dropping dependence on the Lips-
chitz constant. Further, tighter bounds (Haghifam et al., 2020; Zhou et al., 2021; Rodr´ıguez-G´alvez
et al., 2021; Neu, 2021; Hellstr¨om & Durisi, 2021) are proposed based on conditional mutual infor-
mation (Steinke & Zakynthinou, 2020; Gr¨unwald et al., 2021; Hellstr¨om & Durisi, 2020). Due to
space limitations, an extended discussion of the related work is deferred to Appendix A.

2 GENERALIZATION BOUNDS WITH EXPECTED STABILITY

In the setting of statistical learning, there is an instance space Z, a hypothesis space W, and a loss
function (cid:96) : W ×Z (cid:55)→ R+. Let D be an unknown distribution of Z and let S ∼ Dn be n i.i.d. draws
from D. For any speciﬁc hypothesis w ∈ W, the population and empirical loss are respectively
given by LD(w) (cid:44) Ez∼D[(cid:96)(w, z)], and LS(w) (cid:44) 1
i=1 (cid:96)(w, zi). For any distribution P over
n
the hypothesis space, we respectively denote the expected population and empirical loss as
LS(P ) (cid:44) 1
Ew∼P [(cid:96)(w, zi)] .
n

LD(P ) (cid:44) Ez∼DEw∼P [(cid:96)(w, z)] ,

n
(cid:88)

(cid:80)n

and

(1)

Consider a randomized algorithm A which works with S = {z1, . . . , zn} ∼ Dn and cre-

i=1

2

Under review as a conference paper at ICLR 2022

ates a distribution over the hypothesis space W. For convenience, we will denote the distribu-
tion as A(S). The focus of the analysis is to bound the generalization error of A deﬁned as:
gen(A(S)) (cid:44) LD(A(S)) − LS(A(S)) . We will assume A is permutation invariant, i.e., the or-
dering of samples in S do not modify A(S), an assumption satisﬁed by most learning algorithms.
We will focus on developing bounds for the expectation ES[LD(A(S)) − LS(A(S))], and discuss
high-probability bounds in the Appendix B.

2.1 BOUNDS BASED ON EXPECTED STABILITY

(cid:82)

We start our analysis by noting that the expected generalization error can be upper bounded by
expected stability based on the Hellinger divergence (Sason & Verdu, 2016; Li et al., 2020):
w((cid:112)p(w) − (cid:112)p(cid:48)(w))2dw.
H 2(P (cid:107)P (cid:48)) = 1
2
Proposition 1. Let Sn ∼ Dn and let S(cid:48)
z(cid:48)
n ∼ D. Let A(Sn), A(S(cid:48)
obtained by running randomized algorithm A on Sn, S(cid:48)
EW ∼A(Sn)[(cid:96)2(W, z)] ≤ c0/2, c0 > 0. With H(·, ·) denoting the Hellinger divergence, we have

n be a dataset obtained by replacing zn ∈ Sn with
n) respectively denote the distributions over the hypothesis space W
n. Assume that for Sn ∼ Dn, ∀z ∈ Z,

|ESn∼Dn [LD(A(Sn)) − LS(A(Sn))]| ≤ c0ESn∼Dn Ez(cid:48)

n∼D

(cid:113)

2H 2(cid:0)A(Sn), A(S(cid:48)

n)(cid:1) .

(2)

Remark 2.1. Proposition 1 does not need bounded losses. Just the second moment of (cid:96)(W, z), W ∼
A(Sn), Sn ∼ Dn, ∀z ∈ Z need to be bounded. The assumption is satisﬁed by bounded losses. It is
instructive to compare the assumption to that in recent information theoretic bounds (Haghifam et al.,
2020; Xu & Raginsky, 2017), where one assumes (cid:96)(w, Z), Z ∼ D, ∀w ∈ W to be sub-Gaussian.
Remark 2.2. The bound in Proposition 1 is in terms of expected stability where we consider
ES∼Dn Ez(cid:48)
n∼D[· · · ], an important departure from bounds based on uniform stability (Elisseeff et al.,
2005; Bousquet & Elisseeff, 2002; Mou et al., 2018; Bousquet et al., 2020; Feldman & Vondrak,
2018; 2019) where one considers supS,S(cid:48)∈Z n,|S\S(cid:48)|=1[· · · ]. Replacing sup by E makes the bounds
distribution dependent, and arguably leads to quantitatively tighter bounds.

Note that the Hellinger divergence can be bounded by the KL divergence.
Proposition 2. For any distributions P and P (cid:48), 2H 2(P, P (cid:48)) ≤ min (cid:8)KL(P, P (cid:48)),

(cid:113) 1

2 KL(P, P (cid:48))(cid:9).

2.2 EXPECTED STABILITY OF NOISY STOCHASTIC ITERATIVE ALGORITHMS

We consider a general family of noisy stochastic iterative (NSI) algorithms. Given S ∼ Dn, such
iterative algorithms have two additional sources of randomness in each iteration t: (a) a stochastic
mini-batch of samples SBt, with |SBt| = b, drawn uniformly at random with replacement from
S; and (b) noise ξt suitably included in the iterative update. Given a trajectory of past iterates
W0:(t−1) = w0:(t−1), the new iterate Wt is drawn from a distribution PBt,ξt|w0:(t−1) over W:

Wt ∼ PBt,ξt|w0:(t−1) (W ) .

(3)

We will drop the conditioning w0:(t−1) to avoid clutter in the sequel. Let PT , P (cid:48)
T denote the marginal
distributions over hypotheses w ∈ W after T steps of the algorithm based on Sn, S(cid:48)
n respec-
tively. Further, let P0:(t−1) denote the joint distribution over W0:(t−1) = (W0, . . . , Wt−1), and
let Pt| ≡ PBt,ξt|w0:(t−1) compactly denote the conditional distribution on Wt conditioned on the
trajectory W0:(t−1) = w0:t−1. Following (Negrea et al., 2019; Haghifam et al., 2020; Pensia et al.,
T ) ≤ KL(P0:T (cid:107)P (cid:48)
2018), we use the following chain rule for KL-divergence: KL(PT (cid:107)P (cid:48)
0:T ) =
(cid:80)T
n be size n subsets of ¯S such that
EP0:(t−1) [KL(Pt|(cid:107)P (cid:48)
Sn = {Z1, . . . , Zn−1, Zn} and S(cid:48)
n = {Z1, . . . , Zn−1, Z (cid:48)
n = Zn+1. Let S0 =
{Z1, . . . , Zn−1}. The algorithms we consider use a mini-batch of size b in each iteration uniformly
sampled from n samples. Let the set of all mini-batch index sets be denoted by G. Let the set of all
mini-batch index sets A drawn from S0 be denoted by G0. Note that |G0| = (cid:0)n−1
(cid:1). Let G1 denote
the set of all mini-batch index sets B which includes the last sample, viz. zn for S with mini-batches
(cid:1) = |G|.
and z(cid:48)

t|)]. Let ¯S ∼ Dn+1, and let Sn, S(cid:48)

(cid:1). Also note that |G0| + |G1| = (cid:0)n−1

n. Note that |G1| = (cid:0)n−1

n}, where Z (cid:48)

(cid:1) + (cid:0)n−1

n for S(cid:48)

(cid:1) = (cid:0)n

t=1

b

b−1

b

b−1

b

Following Li et al. (2020), we can bound their conditional KL-divergences KL(Pt|(cid:107)P (cid:48)
t|) in terms
of a Le Cam Style Divergence (LSD). While the classical Le Cam divergence (Sason & Verdu,

3

Under review as a conference paper at ICLR 2022

(where dP denotes the density), our bounds in terms of

2016) is LCD(P (cid:107)P (cid:48)) (cid:44) 1
2

(cid:82) (dP −dP (cid:48))2
dP +dP (cid:48)
)2

t|) (cid:44) (cid:82) (dPBt,ξt −dP (cid:48)

Bt,ξt

dPAt,ξt

LSD(Pt||(cid:107)P (cid:48)
the distribution of Wt for Sn and S(cid:48)
the n-th sample. Putting everything together, we have the following LSD based bound.
Lemma 1. Consider a noisy stochastic iterative algorithms of the form (3) with mini-batch size
b ≤ n/2. Then, with c1 =
2c0 (with c0 as in Proposition 1), we have

, Bt ∈ G1, At ∈ G0. Note that PBt,ξt and P (cid:48)
Bt,ξt
n respectively since the mini-batch SBt of Sn and S(cid:48)

represent
n differs in

√

|ESn [LD(A(Sn))−LSn(A(Sn))]| ≤ c1

b
n

ESn

Ez(cid:48)

n

(cid:118)
(cid:117)
(cid:117)
(cid:117)
(cid:117)
(cid:116)

T
(cid:88)

t=1

E
W0:(t−1)

E
Bt∈G1

E
At∈G0

(cid:16)


(cid:90)



ξt

dPBt,ξt − dP (cid:48)
dPAt,ξt

Bt,ξt

(cid:17)2



dξt


 .

(4)
Remark 2.3. Li et al. (2020) essentially has this result for SGLD and inspired our work. Our proofs
are signiﬁcantly simpler and, more importantly, illustrates applicability to general noisy iterative
algorithms of the form (3), not just SGLD with Gaussian noise as in Li et al. (2020).
Remark 2.4. Note that the bound does not assume the loss to be bounded, depends on expectations
over samples Sn, z(cid:48)
n, trajectories w0:(t−1), and mini-batches Bt, At. Further, the bound depends on
the distribution discrepancy as captured by the expected LSD.
Remark 2.5. The bound seems to worsen with b, the size of the mini-batch. As we shown in Sec-
tion 3, the expected LSD term has a 1
b2 dependence for the Exponential Family Langevin dynamics
(EFLD) models we introduce, so the leading b is neutralized.

3 EXPONENTIAL FAMILY LANGEVIN DYNAMICS

In recent years, considerable advances have been made in establishing generalization bounds for
stochastic gradient Langevin dynamics (SGLD) (Li et al., 2020; Pensia et al., 2018; Negrea et al.,
2019; Haghifam et al., 2020). As an example of NSI algorithms of the form (3), SGLD adds an
(cid:1), where
isotropic Gaussian noise at every step of SGD: wt+1 = wt − ηt∇(cid:96)(wt, SBt) + N (cid:0)0, σ2
Id
∇(cid:96)(wt, SBt) is the stochastic gradient on mini-batch Bt, ηt is the step size, and σ2
t is noise vairance.
In this paper, we introduce a substantial generalization of SGLD called Exponential Family
Langevin Dynamics (EFLD) which uses general exponential family noise in noisy iterative updates
of the form (3). In addition to being a mathematical generalization of the popular SGLD, the pro-
posed EFLD provides ﬂexibility to use noise gradient algorithms with different representation of the
gradient, e.g., Bernoulli noise for Sign-SGD, discrete distribution for quantized or ﬁnite precision
SGD, etc. (Canonne et al., 2020; Alistarh et al., 2017; Jiang & Agrawal, 2018; Yang et al., 2019).

t

j=1 exp(ξjθjα − ψj(θjα))π0(ξj) , where ξ is the sufﬁcient statistic, ψ(θα) = (cid:80)p

3.1 EXPONENTIAL FAMILY LANGEVIN DYNAMICS (EFLD)
Exponential families (Barndorff-Nielsen, 2014; Brown, 1986; Wainwright & Jordan, 2008) consti-
tute a large family of parametric distributions which include Gaussian, Bernoulli, gamma, Pois-
son, Dirichlet, etc., as special cases. Exponential families are typically represented in terms of
natural parameters θ, and we consider component-wise independent distributions with scaled nat-
ural parameter θα = θ/α with scaling α > 0, i.e., pψ(ξ, θα) = exp((cid:104)ξ, θα(cid:105) − ψ(θα))π0(ξ) =
(cid:81)p
j=1 ψj(θjα) is
the log-partition function, and π0(ξ) = (cid:81)p
j=1 π0(ξj) is the base measure. Note that α = 1 gives
the canonical form of the exponential family distributions. For general scaling α > 0, for some
cases the base measure π0 may depend on the scaling, i.e., π0,α. A scaling α > 0 is valid as long
as exp((cid:104)ξ, θα(cid:105) is integrable, i.e., (cid:82)
ξ exp((cid:104)ξ, θα(cid:105)π0(ξ)dξ < ∞. Further, ψ is a smooth function by
construction (Barndorff-Nielsen, 2014; Banerjee et al., 2005; Wainwright & Jordan, 2008) and the
smoothness of ψ implies ∇2ψ(θα) ≤ c2I.
Exponential family Langevin dynamics (EFLD) uses noisy stochastic gradient updates similar to
SGLD, but using exponential family noise rather than Gaussian noise as in SGLD. In particular, for
mini-batch SBt, EFLD updates are as follows: with step size ρt > 0

where
pψ(ξ; θBt,αt) = exp((cid:104)ξ, θBt,αt(cid:105)−ψ(θBt,αt))π0(ξ) ,

wt = wt−1 − ρtξt ,

ξt ∼ pψ(ξ; θBt,αt) ,
(cid:44) θBt
αt

θBt,αt

=

∇(cid:96)(wt−1, SBt)
αt

(5)

. (6)

4

Under review as a conference paper at ICLR 2022

For EFLD, the natural parameter θBt,αt at step t is simply a scaled version of the mini-batch gradient
∇(cid:96)(wt−1, SBt). We ﬁrst show that EFLD becomes SGLD when the exponential family is Gaussian,
and becomes a noisy version of sign-SGD (Bernstein et al., 2018a;b) when the exponential family is
Bernoulli over {−1, +1}. More details and examples are in Appendix C.1.
Example 3.1 (SGLD). SGLD uses scaled Gaussian noise with ψ(θ) = (cid:107)θ(cid:107)2
so that pψ(ξ; θBt,αt) = N (θBt, α2
t
distributed as N (ρtθBt, ρ2
the SGLD update: wt = wt−1 − ηt∇(cid:96)(wt−1, SBt) + N (cid:0)0, σ2
Example 3.2 (Noisy Sign-SGD). By taking ρt = ηt and component-wise ξj ∈ {−1, 1}, π0(ξj) =
1, ψ(θ) = log(exp(−θ) + exp(θ)) in exponential family update equation (5), the j-th component of
exp(ξj θBt,αt,j )
exponential family distribution pψ(ξ; θBt,αt) becomes pθBt,αt,j (ξj) =
exp(−θBt,αt,j )+exp(θBt,αt,j ) .
Thus, the EFLD update reduces to a noisy version of Sign-SGD: wt = wt−1 − ηtξt, ξt,j ∼
pθBt,αt,j (ξj), j ∈ [d], where θBt,αt = ∇(cid:96)(wt−1, SBt)/αt is the scaled mini-batch gradient.

2/2, αt = (cid:112)σt/ηt,
ηtσt, the update (5) based on ρtξt is
Id). Thus the EFLD update reduces to
t

Id) = N (ηt∇(cid:96)(wt−1, SBt), σ2
t

Id). By taking ρt =

t α2
t

(cid:1) .

Id

√

3.2 EXPECTED STABILITY OF EXPONENTIAL FAMILY LANGEVIN DYNAMICS

From Lemma 1, conditioned on a trajectory w0:(t−1), mini-batches SBt, SAt, we can get gener-
alization bound by suitably bounding the Le Cam Style Divergence (LSD) given by: IAt,Bt =
(cid:82)

dξt. For EFLD, the density functions dPBt,ξt are exponential family densi-

Bt,ξt)2

(dPBt,ξt −dP (cid:48)
dPAt,ξt

ξt

ties pψ(ξ; θBt,αt) as in (5)-(6), and we have the following bound on the per step LSD:
Theorem 1. For a given set ¯S ∼ Dn+1 and wt−1 at iteration (t − 1), let ∆t|wt−1( ¯S) =
maxz,z(cid:48)∈ ¯S (cid:107)∇(cid:96) (wt−1, z) − ∇(cid:96) (wt−1, z(cid:48))(cid:107)2 . Further, for a c2-smooth log-partition function ψ,
let the scaling αt|wt−1 be data-dependent such that α2
(Sn+1). Then, we have

≥ 8c2∆2

t|wt−1

t|wt−1

IAt,Bt ≤ 5c2(cid:107)θBt,αt − θB(cid:48)

t,αt(cid:107)2

2 =

5c2

(cid:104)(cid:13)
(cid:13)∇(cid:96) (wt−1, SBt) − ∇(cid:96) (cid:0)wt−1, S(cid:48)

(cid:105)

(cid:1)(cid:13)
2
(cid:13)
2

,

(7)

Bt

2α2
only differ at samples zn and z(cid:48)

t|wt−1

Note that SBt and S(cid:48)
n. The above bound can now be directly
Bt
applied to Lemma 1 to get expected stability based generalization bounds for any EFLD algorithm.
Theorem 2. Consider an exponential family Langevin dynamics (EFLD) algorithm of the form (5)-
(6) with a c2-smooth log-partition function ψ. Then, for mini-batch size b ≤ n/2, with c = c0
5c2
t| ≥ 8c2∆2
(with c0 as in Lemma 1) and α2
t|(Sn+1) (as in Theorem 1, with the conditioning on wt−1
hidden to avoid clutter), we have
(cid:118)
(cid:117)
(cid:117)
(cid:116)

(cid:107)∇(cid:96) (wt−1, zn) − ∇(cid:96) (wt−1, z(cid:48)

|ES[LD(A(S)) − LS(A(S))]| ≤ c

T
(cid:88)

√

(cid:104)

(cid:105)

.

n)(cid:107)2

2

1
n

E
Sn+1

E
W0:(t−1)

1
α2
t|

t=1

(cid:80)

(8)
Remark 3.1. Theorem 2 captures the generalization error of SGLD, which is a special case of
EFLD. Our bound has the same dependence on n, T , step size ηt as the bound in Li et al.
(2020). However, our bound is numerically sharper because we replace the gradient norms, i.e.,
1
z∈S (cid:107)(cid:96)(wt, z)(cid:107) in Li et al. (2020) and with gradient discrepancy (cid:107)∇(cid:96)(wt, z) − ∇(cid:96)(wt, z(cid:48))(cid:107),
n
which is quantitatively smaller than gradient norms as we show in the experiment section. The
bound in Negrea et al. (2019) depends on gradient incoherence which is empirically smaller than
n, which is
gradient discrepancy as observed in the experiment section, their bound depends on 1/
worse than the 1/n dependence in our bound.
Remark 3.2. EFLD can be extended to work with anisotropic noise by using θBt,αt =
∇(cid:96)(wt−1, SBt) (cid:11) αt in (6) where αt ∈ Rp determines different scaling for each dimension and
(cid:11) denotes Hadamard division. Theorems 1 and 2 can be extended to such anisotropic noise by using
α-scaled norms for the gradient discrepancy, i.e., (cid:107)g − g(cid:48)(cid:107)2
Remark 3.3. The condition on αt is a data-dependent quantity, which can be computed along the
training process. It gives much more benign condition of the step size compared to those in the
related work (Mou et al., 2018; Li et al., 2020, Hardt et al. 2016), which require step size being
bounded by σt/L. However, the step sizes in Theorem 2 need to be bounded by σt/∆t( ¯S), which
is considerably more relaxed since ∆t( ¯S) is much smaller than Lipschitz constant L, which is a
uniform bound over the whole parameter space. Also, usually one would expect ∆t( ¯S) to decrease
as training proceeds since the gradients shrink as the loss function being minimized. Thus, the
constraint on step size does not require the step sizes to be as small as σt/L.

2,α = (cid:80)

j(gj − g(cid:48)

j)2/α2
j .

√

5

Under review as a conference paper at ICLR 2022

3.3 PROOF SKETCHES OF MAIN RESULTS: THEOREMS 1 AND 2

We focus on Theorem 1. To avoid clutter, we drop the subscript t for the analysis and note that
the analysis holds for any step t. When the density dPB,ξ = pψ(ξ; θB,α), by mean-value theorem,
pψ(ξ; ˜θB,α)(cid:105), for some
for each ξ, we have pψ(ξ; θB,α) − pψ(ξ; θB(cid:48),α) = (cid:104)θB,α − θB,α, ∇ ˜θB,α
˜θB,α = γξθB,α + (1 − γξ)θ(cid:48)

B,α where γξ ∈ [0, 1]. Then,

IA,B =

(cid:90)

ξ

(cid:0)pψ(ξ; θB,α) − pψ(ξ; θB(cid:48),α)(cid:1)2
pψ(ξ; θA,α)

dξ =

(cid:90)

ξ

(cid:104)θB,α − θ(cid:48)

B,α, ξ − ∇ ˜θB,α

ψ(ξ; ˜θB,α)(cid:105)2 p2

ψ(ξ; ˜θB,α)

pψ(ξ; θA,α)

dξ ,

since pψ(ξ; ˜θB,α) = exp((cid:104)ξ, ˜θB,α(cid:105) − ψ( ˜θB,α))π0(ξ).
Handling Distributional Dependence of ˜θB. Note that we cannot proceed with the analysis with
the density term depending on ˜θB,α since ˜θB,α depends on ξ. So, we ﬁrst bound the density term
depending on ˜θB,α in terms of exponential family densities with parameters θB,α and θB,α using
c2-smoothness of ψ.
Lemma 2. For some γξ ∈ [0, 1], ˜θB,α = γξθB,α + (1 − γξ)θ(cid:48)

B,α, we have

(cid:105)
(cid:104)
(cid:104)ξ, ˜θB,α(cid:105) − ψ( ˜θB,α)

exp

max (cid:0)exp (cid:2)(cid:104)ξ, θB,α(cid:105) − ψ(θB,α)(cid:3), exp [(cid:104)ξ, θB(cid:48),α(cid:105) − ψ(θB(cid:48),α)](cid:1) ≤ exp (cid:2)c2(cid:107)θB,α − θB(cid:48),α(cid:107)2

2

(cid:3) .

ψ(ξ, ˜θB,α)/pψ(ξ; θA,α). By
Bounding the Density Ratio. Next we focus on the density ratio p2
Lemma 2, it sufﬁces to focus on p2
ψ(ξ, θB,α)/pψ(ξ; θA,α) or the equivalent term for θB(cid:48),α. We show
that the density ratio can be bounded by another exponential family with parameters (2θB,α −θA,α).

Lemma 3. For any ξ, we have

exp [(cid:104)ξ, 2θB,α(cid:105) − 2ψ(θB,α)]
exp [(cid:104)ξ, θA,α(cid:105) − ψ(θA,α)]

≤ exp (cid:2)2c2(cid:107)θB,α − θA,α(cid:107)2

2

(cid:3) exp [(cid:104)ξ, (2θB,α − θA,α(cid:105) − ψ(2θB,α − θA,α)] .

The analysis for the term p2

ψ(ξ, θB(cid:48),α)/pψ(ξ; θA,α) is exactly the same.

(cid:82)

the analysis needs to bound an integral

Ignoring multiplicative terms which do not depend on ξ for the
Bounding the Integral.
(cid:104)θB,α − θ(cid:48)
B,α, ξ −
term of the form
moment,
∇ψ(ξ; ˜θB,α)(cid:105)2 pψ(ξ; 2θB,α − θA,α)dξ, and a similar term with p2
ψ(ξ; 2θB(cid:48),α − θA,α). First,
note that ∇ψ(ξ; ˜θB,α) = ˜µB,α, the expectation parameter for pψ(ξ; ˜θB,α) Wainwright & Jordan
(2008); Banerjee et al. (2005). The integral, however, is with respect to pψ(ξ; 2θB,α − θA,α).
We handle this discrepancy by using ξ − ∇ψ(ξ; ˜θB,α) = (ξ − E[ξ]) + (E[ξ] − ∇ψ(ξ; ˜θB,α)),
and decomposing as sum-of-squares. Quadratic form for the ﬁrst term yields the covariance
E[(ξ − E[ξ])(ξ − E[ξ])T ] = ∇2ψ(θ2θB,α−θA,α ) ≤ c2I, by smoothness. The second term de-
pends on the difference of gradients ∇ψ(2θB,α − θA,α) − ∇ψ( ˜θB,α) which, using smoothness and
additional analysis, can be bounded by the norm of (θB,α −θA,α). All the pieces can be put together
to get the bound in Theorem 1, which when used in Lemma 1 yields Theorem 2.

ξ

3.4 OPTIMIZATION GUARANTEES FOR EFLD
We now establish optimization guarantees for two examples of EFLD, i.e., Noisy Sign-SGD with
Bernoulli noise over {−1, +1} and SGLD with Gaussian noise.

Noisy Sign-SGD. For mini-batch Bt and scaling αt, mini-batch Noisy Sign-SGD updates the pa-
rameters as wt = wt−1 − ηtξt, where each component j ∈ [d]

ξt,j ∼ pθBt,αt,j (x) =

exp(xθBt,αt,j)
exp(−θBt,αt,j) + exp(θBt,αt,j)

, x ∈ {−1, +1}

(9)

where θBt,αt = ∇(cid:96)(wt−1, SBt)/αt is the scaled mini-batch gradient. The full-batch version uses
parameters EBt[θBt,αt] = ∇LS(wt−1) For the optimization analysis, we assume that the loss is
smooth and mini-batch gradients are unbiased, symmetric, and sub-Gaussian.

6

Under review as a conference paper at ICLR 2022

√

i Ki(wi − w(cid:48)
the mini-batch gradient ∇(cid:96)(wt−1, SBt) is (a) unbiased,

Assumption 1. The loss function LS satisﬁes: for all w and w(cid:48), for some non-negative constant
(cid:126)K := [K1, . . . , Kd], we have LS(w) ≤ LS(w(cid:48)) + ∇LS(w(cid:48))T (w − w(cid:48)) + 1
2
i.e.,
Assumption 2. Given wt−1,
EBt|wt−1∇(cid:96)(wt−1, SBt) = ∇LS(wt−1);
the density p(x) of x ≡
∇(cid:96)(wt−1, SBt) is symmetric around its expectation LS(wt−1): p(x) = p(2∇LS(wt−1) − x) and
(c) sub-Gaussian, i.e., for any λ > 0, any v s.t. (cid:107)v(cid:107)2 = 1, EBt|wt−1 exp λ(cid:104)v, ∇(cid:96)(wt−1, SBt) −
∇LS(wt−1)(cid:105) ≤ exp(λ2κ2
Based on the assumptions, we have the following optimization guarantee for mini-batch noisy Sign-
SGD. We defer the optimization guarantee for full-batch noisy Sign-SGD to Appendix D.

t /2) for some constant κt > 0.

(b) symmetric,

i)2.

i.e.,

(cid:80)

Theorem 3. Under Assumption 1 and 2, for mini-batch noisy Sign-SGD with step size ηt = 1/
αt satisfying c ≥ αt ≥ max[
w0

T ,
2κt, 4(cid:107)∇LS(wt−1)(cid:107)∞], we have for any S and any initialization

√

(cid:35)

(cid:107)∇LS(wt)(cid:107)2
2

≤

LS(w0) − LS(w∗) +

(cid:107) (cid:126)K(cid:107)1

,

(10)

(cid:19)

1
2

(cid:18)

4c
√
T

(cid:34)

E

1
T

T
(cid:88)

t=1

where the expectation is taken over the randomness of algorithm.

SGLD. We acknowledge that the following optimization result of SGLD exists in various forms, as
noisy gradient descent algorithms have been studied in literature such as differential privacy, where
SGLD can be viewed as DP-SGD (Bassily et al., 2014; Wang & Xu, 2019) and the proof technique
boils down to bounding the stochastic variance of the noisy gradient (Shamir & Zhang, 2013).
Theorem 4. Under Assumption 1 and 2, with Ki = K, ∀i ∈ [d], for any S, SGLD (EFLD with step
size ρt =

ηtσt, αt = (cid:112)σt/ηt), |Bt| = b, and ηt = 1√

, can achieve

√

T

1
T

T
(cid:88)

t=1

E(cid:107)∇LS(wt)(cid:107)2 ≤ O

(cid:19)

(cid:18) 1
√
T

(cid:32)

+ O

K

p (cid:80)T

t=1 α4
t + log T
√
T

(cid:33)

,

(11)

where the expectation is over the randomness of the algorithm.
The error rate of SGLD depends on the noise variance αt. One can choose a decaying noise variance
√
such as αt = 1/ 4
T ). We
note that similar to the optimization guarantees of DP-SGD, the convergence rate depends on the
dimension of the gradient p due to the isotropic Gaussian noise. Special noise structures such as
anisotropic noise that aligned with the gradient structure can reduce the dependence on dimension
(Kairouz et al., 2020; Zhang et al., 2021; Asi et al., 2021; Zhou et al., 2020).

t to guarantee the convergence. Then the rate will become O(log T /

√

4 EXPERIMENTS

In this section, we conduct a series of experiments to evaluate our generalization error bounds. For
SGLD, we aim to compare the proposed bound in Theorem 2 with existing bounds in Li et al. (2020),
Negrea et al. (2019), and Rodr´ıguez-G´alvez et al. (2021) for various datasets. Note that the bound
presented in Rodr´ıguez-G´alvez et al. (2021) is an extension of that in Haghifam et al. (2020) from
full-batch setting to mini-batch setting . We also evaluate the optimization performance of proposed
Noisy Sign-SGD by comparing it with the original sign-SGD (Bernstein et al., 2018a) and present
the corresponding generalization bound in Theorem 2.

The details of our model architectures, learning rate scheduling, hyper-parameter selections and
additional experimental results can be found in Appendix E. We acknowledge that we did not achieve
the state-of-the art predictive performance, mainly due to the simplicity of our model architectures.
With more complex model and further tuning, the prediction results could be improved.

4.1 STOCHASTIC GRADIENT LANGEVIN DYNAMICS
Comparison with existing work. We have derived theoretical generalization error bounds that
n)(cid:107)2
depend on the data-dependent quantity gradient discrepancy, i.e., (cid:107)∇(cid:96) (wt, zn) − ∇(cid:96) (wt, z(cid:48)
2.
Existing bounds in Li et al. (2020) and Negrea et al. (2019) have also improved the Lipschitz constant
in Mou et al. (2018) to a data-dependent quantity. As shown in Figure 1 (a)-(d), by combining with
the empirical training error, all four generalization error bounds can be used to bound the empirical
test error, but our bound is able to generate a much tighter upper bound. Such difference is mainly
due to the fact that we replace the squared gradient norm in Li et al. (2020), the squared norm of

7

Under review as a conference paper at ICLR 2022

(a) MNIST, α2

t ≈ 0.1

(b) CIFAR-10, α2

t ≈ 0.1

(c) Fashion, α2

t ≈ 0.1

(d) Fashion, α2

t ≈ 0.01

(e) MNIST, α2

t ≈ 0.1

(f) CIFAR-10, α2

t ≈ 0.1

(g) Fashion, α2

t ≈ 0.1

(h) Fashion, α2

t ≈ 0.01

(i) MNIST, α2

t ≈ 0.1

(j) CIFAR-10, α2

t ≈ 0.1

(k) Fashion, α2

t ≈ 0.1

(l) Fashion, α2

t ≈ 0.01

Figure 1: Numerical results for training CNN using SGLD (σt = (cid:112)2ηt/βt) on MNIST, Fashion-
MNIST and CIFAR-10. X-axis shows the number of training epochs. (a)-(d) shows our bound is
non-vacuous and can be used to bound the empirical test error. (e)-(h) compare our bound with
the existing bounds and show the effect on α2
(i)-(l) show the key factors in each bound, i.e.,
t .
the squared gradient norm in Li et al. (2020), the gradient incoherence in Negrea et al. (2019),
the two-sample incoherence in Rodr´ıguez-G´alvez et al. (2021), and the gradient discrepancy in our
bound. Our bounds are numerically sharper than existing bounds, and larger α2
t leads to tighter
generalization bounds which is consistent with the theoretical analysis.

gradient incoherence in Negrea et al. (2019), and that of two-sample incoherence in Rodr´ıguez-
G´alvez et al. (2021) with the gradient discrepancy. Results in Figure 1 (e)-(h) show that our bounds
are much sharper than those of Li et al. (2020) because our gradient discrepancy (Figure 1 (i)-(l))
is usually 2-4 order of magnitude smaller than the squared gradient norms appeared in Li et al.
(2020). Our bounds are also sharper than those of Negrea et al. (2019) and Rodr´ıguez-G´alvez et al.
(2021) due to an improved dependence on n from an order of 1/
n to 1/n. Note that, even though
the gradient incoherence in Negrea et al. (2019) is about 1 to 2 order of magnitude smaller than
the gradient discrepancy for simple problems such as MNIST and Fashion-MNIST, the difference
between the gradient incoherence and our gradient discrepancy reduces as the problem becomes
harder (see results for CIFAR-10 in Figure 1(j)).

√

Effect of Randomness. Motivated by Zhang et al. (2017), we train CNN with SGLD on a smaller
subset of MNIST dataset (n = 10000) with randomly corrupted labels. The corruption fraction
varies from 0% (without label corruption) to 60%. As shown in Figure 2 (d), for long enough training
time, all experiments with different level of label randomness can achieve almost zero training error.
However, the one with higher level of randomness has higher generalization/test error (Figure 2 (a)
dashed lines). Our generalization bound also becomes larger as the randomness increases since the
corresponding gradient discrepancy increases.

4.2 NOISY SIGN-SGD

Optimization. Figure 3 (a)-(d) show the training dynamics of Noisy Sign-SGD under various se-
lections of αt. As αt → 0, Noisy Sign-SGD matches both the optimization trajectory as well as the
ﬁnal test accuracy of the original Sign-SGD (Bernstein et al., 2018a). However, as αt increases, the
probability of getting 1 approaches 0.5, and ξt approximates a uniform distribution. As a result, the
corresponding Noisy Sign-SGD still converges, but the generalization performance is much worse.

8

Under review as a conference paper at ICLR 2022

(a) Bounded Test Error

(b) Our Bound

(c) Gradient Discrepancy

(d) Training Error

Figure 2: Numerical results for training CNN using SGLD (σt = 0.2ηt) on a subset of MNIST (n =
10000) with different randomness on labels. (a) demonstrates that, as the randomness increases, the
empirical test error (dashed lines) increases but still can be bounded by our generalization bound by
combining the empirical training error (solid lines). (b) presents our bound in Theorem 2. (c) shows
the gradient discrepancy (cid:107)∇(cid:96) (wt, zn) − ∇(cid:96) (wt, z(cid:48)
2. (d) plots the training error. The gradient
discrepancy increases as randomness increases, so does our generalization bound.

n)(cid:107)2

(a) CNN, MNIST

(b) CNN, Fashion

(c) ResNet-18, CIFAR10 (d) ResNet-18, CIFAR100

(e) MNIST, αt = 1

(f) MNIST, αt = 0.01

(g) Fashion, αt = 1

(h) Fashion, αt = 0.01

Figure 3: (a)-(d) show the training dynamics of CNN on MNIST and Fashion-MNIST, and ResNet-
18 on CIFAR-10 and CIFAR-100 using noisy sign-SGD with different scaling αt. Legends indicate
the choice of αt and the numbers in brackets are test errors at convergence. As αt → 0, Nosiy
sign-SGD matches both the optimization trajectory as well as the ﬁnal test accuracy of the original
sign-SGD (Bernstein et al., 2018a). (e)-(f) show that empirical test error can be bounded by our
bound and the corresponding training error. The larger αt is the sharper our bound is.

Generalization Bound. Figure 3(e)-(f) show that our bound successfully bounds the empirical test
error. The larger αt is the sharper the upper bound is. However, larger αt would slow down and
adversely affect the optimization, e.g., Figure 3 (a)-(d) blue and orange lines. In practice, one needs
to balance the optimization error and generalization by choosing a suitable scaling αt.

5 CONCLUSIONS

Inspired by recent advances in stability based and information theoretic approaches to generalization
bounds (Mou et al., 2018; Pensia et al., 2018; Negrea et al., 2019; Li et al., 2020; Haghifam et al.,
2020), we have presented a framework for developing such bounds based on expected stability for
noisy stochastic iterative (NSI) learning algorithms. We have also introduced Exponential Family
Langevin Dynamics (EFLD), a family of noisy gradient descent algorithms based on exponential
family noise, including SGLD and Noisy Sign-SGD as two special cases. We have developed an
expected stability based generalization bound applicable to any EFLD algorithm with a O(1/n)
sample dependence and a dependence on gradient incoherence, rather than gradient norms. Further,
we have provided optimization guarantees for EFLD and establish such results for Noisy Sign-SGD
and SGLD. Our experiments on various benchmarks illustrate that our bounds are non-vacuous and
quantitatively much sharper than existing bounds (Li et al., 2020; Negrea et al., 2019).

9

Under review as a conference paper at ICLR 2022

REFERENCES

Dan Alistarh, Demjan Grubic,

Qsgd:
Communication-efﬁcient sgd via gradient quantization and encoding. In I. Guyon, U. V. Luxburg,
S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural
Information Processing Systems 30, pp. 1709–1720. Curran Associates, Inc., 2017.

Jerry Li, Ryota Tomioka, and Milan Vojnovic.

Hilal Asi, John Duchi, Alireza Fallah, Omid Javidbakht, and Kunal Talwar. Private adaptive gradient
methods for convex optimization. In International Conference on Machine Learning, pp. 383–
392. PMLR, 2021.

Arindam Banerjee, Srujana Merugu, Inderjit S Dhillon, and Joydeep Ghosh. Clustering with breg-

man divergences. Journal of machine learning research, 6(10), 2005.

Ole Barndorff-Nielsen. Information and exponential families: in statistical theory. John Wiley &

Sons, 2014.

Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Efﬁcient
In 2014 IEEE 55th Annual Symposium on Foundations of

algorithms and tight error bounds.
Computer Science, pp. 464–473. IEEE, 2014.

Raef Bassily, Vitaly Feldman, Kunal Talwar, and Abhradeep Guha Thakurta. Private stochastic
convex optimization with optimal rates. Advances in neural information processing systems,
2019.

Raef Bassily, Vitaly Feldman, Crist´obal Guzm´an, and Kunal Talwar. Stability of stochastic gradient
descent on nonsmooth convex losses. Advances in Neural Information Processing Systems, 33,
2020.

Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar.
In International Conference on

signsgd: Compressed optimisation for non-convex problems.
Machine Learning, pp. 560–569. PMLR, 2018a.

Jeremy Bernstein, Jiawei Zhao, Kamyar Azizzadenesheli, and Anima Anandkumar. signsgd with
majority vote is communication efﬁcient and fault tolerant. In International Conference on Learn-
ing Representations, 2018b.

St´ephane Boucheron, G´abor Lugosi, and Pascal Massart. Concentration inequalities: A nonasymp-

totic theory of independence. Oxford university press, 2013.

Olivier Bousquet and Andr´e Elisseeff. Stability and generalization. Journal of Machine Learning

Research, 2:499–526, 2002.

Olivier Bousquet, Yegor Klochkov, and Nikita Zhivotovskiy. Sharper bounds for uniformly stable

algorithms. In Conference on Learning Theory, pp. 610–626. PMLR, 2020.

Lawrence D Brown. Fundamentals of statistical exponential families: with applications in statistical

decision theory. Ims, 1986.

Yuheng Bu, Shaofeng Zou, and Venugopal V Veeravalli. Tightening mutual information based
bounds on generalization error. In 2019 IEEE International Symposium on Information Theory
(ISIT), pp. 587–591. IEEE, 2019.

Mark Bun, Cynthia Dwork, Guy N Rothblum, and Thomas Steinke. Composable and versatile
privacy via truncated cdp. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory
of Computing, pp. 74–86, 2018.

Cl´ement L Canonne, Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential

privacy. In NeurIPS, 2020.

Xiangyi Chen, Tiancong Chen, Haoran Sun, Zhiwei Steven Wu, and Mingyi Hong. Distributed
training with heterogeneous data: Bridging median-and mean-based algorithms. arXiv preprint
arXiv:1906.01736, 2019.

10

Under review as a conference paper at ICLR 2022

Luc Devroye and Terry Wagner. Distribution-free inequalities for the deleted and holdout error

estimates. IEEE Transactions on Information Theory, 25(2):202–207, 1979.

Andre Elisseeff, Theodoros Evgeniou, Massimiliano Pontil, and Leslie Pack Kaelbing. Stability of

randomized learning algorithms. Journal of Machine Learning Research, 6(1), 2005.

Vitaly Feldman and Jan Vondrak. Generalization bounds for uniformly stable algorithms. In Pro-
ceedings of the 32nd International Conference on Neural Information Processing Systems, pp.
9770–9780, 2018.

Vitaly Feldman and Jan Vondrak. High probability generalization bounds for uniformly stable al-
gorithms with nearly optimal rate. In Conference on Learning Theory, pp. 1270–1279. PMLR,
2019.

Peter Gr¨unwald, Thomas Steinke, and Lydia Zakynthinou. Pac-bayes, mac-bayes and condi-
tional mutual information: Fast rate bounds that handle general vc classes. arXiv preprint
arXiv:2106.09683, 2021.

Mahdi Haghifam, Jeffrey Negrea, Ashish Khisti, Daniel M Roy, and Gintare Karolina Dziugaite.
Sharpened generalization bounds based on conditional mutual information and an application to
noisy, iterative algorithms. Advances in Neural Information Processing Systems, 2020.

Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic

gradient descent. In International Conference on Machine Learning, pp. 1225–1234, 2016.

Fredrik Hellstr¨om and Giuseppe Durisi. Generalization bounds via information density and condi-
tional information density. IEEE Journal on Selected Areas in Information Theory, 1(3):824–839,
2020.

Fredrik Hellstr¨om and Giuseppe Durisi. Fast-rate loss bounds via conditional information measures
In 2021 IEEE International Symposium on Information

with applications to neural networks.
Theory (ISIT), pp. 952–957. IEEE, 2021.

Peng Jiang and Gagan Agrawal. A linear speedup analysis of distributed deep learning with sparse
and quantized communication. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-
Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 2525–
2536. Curran Associates, Inc., 2018.

Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle

points efﬁciently. In International Conference on Machine Learning, pp. 1724–1732, 2017.

Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M Kakade, and Michael I Jordan. On nonconvex
optimization for machine learning: Gradients, stochasticity, and saddle points. arXiv preprint
arXiv:1902.04811, 2019.

Richeng Jin, Yufan Huang, Xiaofan He, Tianfu Wu, and Huaiyu Dai. Stochastic-sign sgd for feder-

ated learning with theoretical guarantees. arXiv preprint arXiv:2002.10940, 2020.

Peter Kairouz, M´onica Ribero, Keith Rush, and Abhradeep Thakurta. Dimension independence in
unconstrained private erm via adaptive preconditioning. arXiv preprint arXiv:2008.06570, 2020.

Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical Report Vol.

1. No. 4., University of Toronto, 2009.

Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to

document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Jian Li, Xuanyuan Luo, and Mingda Qiao. On generalization error bounds of noisy gradient methods
for non-convex learning. In International Conference on Learning Representations, 2020. URL
https://openreview.net/forum?id=SkxxtgHKPS.

Wenlong Mou, Liwei Wang, Xiyu Zhai, and Kai Zheng. Generalization bounds of sgld for non-
convex learning: Two theoretical viewpoints. In Conference on Learning Theory, pp. 605–638.
PMLR, 2018.

11

Under review as a conference paper at ICLR 2022

Jeffrey Negrea, Mahdi Haghifam, Gintare Karolina Dziugaite, Ashish Khisti, and Daniel M Roy.
Information-theoretic generalization bounds for sgld via data-dependent estimates. In Advances
in Neural Information Processing Systems, 2019.

Gergely Neu. Information-theoretic generalization bounds for stochastic gradient descent. arXiv

preprint arXiv:2102.00931, 2021.

Ankit Pensia, Varun Jog, and Po-Ling Loh. Generalization error bounds for noisy, iterative algo-
rithms. In 2018 IEEE International Symposium on Information Theory (ISIT), pp. 546–550. IEEE,
2018.

David Pollard. A user’s guide to measure theoretic probability. Number 8. Cambridge University

Press, 2002.

Borja Rodr´ıguez-G´alvez, Germ´an Bassi, Ragnar Thobaben, and Mikael Skoglund. On random
subset generalization error bounds and the stochastic gradient langevin dynamics algorithm. In
2020 IEEE Information Theory Workshop (ITW), pp. 1–5. IEEE, 2021.

William H Rogers and Terry J Wagner. A ﬁnite sample distribution-free performance bound for

local discrimination rules. The Annals of Statistics, pp. 506–514, 1978.

Daniel Russo and James Zou. Controlling bias in adaptive data analysis using information theory.

In Artiﬁcial Intelligence and Statistics, pp. 1232–1240. PMLR, 2016.

Igal Sason and Sergio Verdu. f -divergence inequalities. IEEE Transactions on Information Theory,

62, 2016.

Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: Conver-
gence results and optimal averaging schemes. In International conference on machine learning,
pp. 71–79. PMLR, 2013.

Thomas Steinke and Lydia Zakynthinou. Reasoning about generalization via conditional mutual

information. In Conference on Learning Theory, pp. 3437–3452. PMLR, 2020.

Alexandre B Tsybakov. Introduction to nonparametric estimation. Springer Science & Business

Media, 2008.

Martin J Wainwright and Michael Irwin Jordan. Graphical models, exponential families, and vari-

ational inference. Now Publishers Inc, 2008.

Di Wang and Jinhui Xu. Differentially private empirical risk minimization with smooth non-convex
loss functions: A non-stationary view. In Proceedings of the AAAI Conference on Artiﬁcial Intel-
ligence, volume 33, pp. 1182–1189, 2019.

Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics.

In

International Conference on Machine Learning, ICML ’11, pp. 681–688, 2011.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmark-

ing machine learning algorithms, 2017.

Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability of learn-
ing algorithms. Advances in Neural Information Processing Systems, 2017:2525–2534, 2017.

Guandao Yang, Tianyi Zhang, Polina Kirichenko, Junwen Bai, Andrew Gordon Wilson, and Christo-
pher De Sa. Swalp: Stochastic weight averaging in low-precision training. 36th International
Conference on Machine Learning (ICML), 2019.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding
deep learning requires rethinking generalization. In 5th International Conference on Learning
Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings.
OpenReview.net, 2017. URL https://openreview.net/forum?id=Sy8gdB9xx.

Huanyu Zhang, Ilya Mironov, and Meisam Hejazinia. Wide network learning with differential pri-

vacy. arXiv preprint arXiv:2103.01294, 2021.

12

Under review as a conference paper at ICLR 2022

Ruida Zhou, Chao Tian, and Tie Liu. Individually conditional individual mutual information bound
on generalization error. In 2021 IEEE International Symposium on Information Theory (ISIT),
pp. 670–675. IEEE, 2021.

Yingxue Zhou, Steven Wu, and Arindam Banerjee. Bypassing the ambient dimension: Private sgd
with gradient subspace identiﬁcation. In International Conference on Learning Representations,
2020.

13

