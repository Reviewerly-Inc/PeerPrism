Under review as a conference paper at ICLR 2021

ADAPTIVE GRADIENT METHODS CONVERGE FASTER
WITH OVER-PARAMETERIZATION
(AND YOU CAN DO A LINE-SEARCH)

Anonymous authors
Paper under double-blind review

ABSTRACT

Adaptive gradient methods are typically used for training over-parameterized
models capable of exactly ﬁtting the data; we thus study their convergence in this
interpolation setting. Under an interpolation assumption, we prove that AMSGrad
with a constant step-size and momentum can converge to the minimizer at the
faster O(1/T ) rate for smooth, convex functions. Furthermore, in this setting, we
show that AdaGrad can achieve an O(1) regret in the online convex optimization
framework. When interpolation is only approximately satisﬁed, we show that
constant step-size AMSGrad converges to a neighbourhood of the solution. On the
other hand, we prove that AdaGrad is robust to the violation of interpolation and
converges to the minimizer at the optimal rate. However, we demonstrate that even
for simple, convex problems satisfying interpolation, the empirical performance of
these methods heavily depends on the step-size and requires tuning. We alleviate
this problem by using stochastic line-search (SLS) and Polyak’s step-sizes (SPS)
to help these methods adapt to the function’s local smoothness. By using these
techniques, we prove that AdaGrad and AMSGrad do not require knowledge
of problem-dependent constants and retain the convergence guarantees of their
constant step-size counterparts. Experimentally, we show that these techniques help
improve the convergence and generalization performance across tasks, from binary
classiﬁcation with kernel mappings to classiﬁcation with deep neural networks.

1

INTRODUCTION

Adaptive gradient methods such as AdaGrad (Duchi et al., 2011), RMSProp (Tieleman & Hinton,
2012), AdaDelta (Zeiler, 2012), Adam (Kingma & Ba, 2015), and AMSGrad (Reddi et al., 2018)
are popular optimizers for training deep neural networks (Goodfellow et al., 2016). These methods
scale well and exhibit good performance across problems, making them the default choice for many
machine learning applications. Theoretically, these methods are usually studied in the non-smooth,
online convex optimization setting (Duchi et al., 2011; Reddi et al., 2018) with recent extensions to
the strongly-convex (Mukkamala & Hein, 2017; Wang et al., 2020; Xie et al., 2020) and non-convex
settings (Li & Orabona, 2019; Ward et al., 2019; Zhou et al., 2018; Chen et al., 2019; Wu et al.,
2019; D´efossez et al., 2020; Staib et al., 2019). An online–batch reduction gives guarantees similar to
stochastic gradient descent (SGD) in the ofﬂine setting (Cesa-Bianchi et al., 2004; Hazan & Kale,
2014; Levy et al., 2018).

However, there are several discrepancies between the theory and application of these methods.
Although the theory advocates for using decreasing step-sizes for Adam, AMSGrad and its vari-
ants (Kingma & Ba, 2015; Reddi et al., 2018), a constant step-size is typically used in practice (Paszke
et al., 2019). Similarly, the standard analysis of these methods assumes a decreasing momentum
parameter, however, the momentum is ﬁxed in practice. On the other hand, AdaGrad (Duchi et al.,
2011) has been shown to be “universal” as it attains the best known convergence rates in both the
stochastic smooth and non-smooth settings (Levy et al., 2018), but its empirical performance is rather
disappointing when training deep models (Kingma & Ba, 2015). Improving the empirical performance
was indeed the main motivation behind Adam and other methods (Tieleman & Hinton, 2012; Zeiler,
2012) that followed AdaGrad. Although these methods have better empirical performance, they are
not guaranteed to converge to the solution with a constant step-size and momentum parameter.

1

Under review as a conference paper at ICLR 2021

Another inconsistency is that although the standard theoretical results are for non-smooth functions,
these methods are also extensively used in the easier, smooth setting. More importantly, adaptive gra-
dient methods are generally used to train highly expressive, large over-parameterized models (Zhang
et al., 2017; Liang & Rakhlin, 2018) capable of interpolating the data. However, the standard theoreti-
cal analyses do not take advantage of these additional properties. On the other hand, a line of recent
work (Schmidt & Le Roux, 2013; Jain et al., 2018; Ma et al., 2018; Liu & Belkin, 2020; Cevher &
V˜u, 2019; Vaswani et al., 2019a;b; Wu et al., 2019; Loizou et al., 2020) focuses on the convergence
of SGD in this interpolation setting. In the standard ﬁnite-sum case, interpolation implies that all
the functions in the sum are minimized at the same solution. Under this additional assumption, these
works show SGD with a constant step-size converges to the minimizer at a faster rate for both convex
and non-convex smooth functions.

In this work, we aim to resolve some of the discrepancies in the theory and practice of adaptive
gradient methods. To theoretically analyze these methods, we consider a simplistic setting - smooth,
convex functions under interpolation. Using the intuition gained from theory, we propose better
techniques to adaptively set the step-size for these methods, dramatically improving their empirical
performance when training over-parameterized models.

1.1 BACKGROUND AND CONTRIBUTIONS

Constant step-size. We focus on the theoretical convergence of two adaptive gradient methods:
AdaGrad and AMSGrad. For smooth, convex functions, Levy et al. (2018) prove that AdaGrad with
√
a constant step-size adapts to the smoothness and gradient noise, resulting in an O(1/T + ζ/
T )
convergence rate, where T is the number of iterations and ζ 2 is a global bound on the variance in the
stochastic gradients. This convergence rate matches that of SGD under the same setting (Moulines
& Bach, 2011). In Section 3, we show that constant step-size AdaGrad also adapts to interpolation
√
and prove an O(1/T + σ/
T ) rate, where σ is the extent to which interpolation is violated. In the
over-parameterized setting, σ2 can be much smaller than ζ 2 (Zhang & Zhou, 2019), implying a faster
convergence. When interpolation is exactly satisﬁed, σ2 = 0, we obtain an O(1/T ) rate, while ζ 2
can still be large. In the online convex optimization framework, for smooth functions, we show that
T ) to O(1) when interpolation is satisﬁed and retains
the regret of AdaGrad improves from O(
its O(
T )-regret guarantee in the general setting (Appendix C.2). Assuming its corresponding
preconditioner remains bounded, we show that AMSGrad with a constant step-size and constant
momentum parameter also converges at the rate O(1/T ) under interpolation (Section 4). However,
unlike AdaGrad, it requires speciﬁc step-sizes that depend on the problem’s smoothness. More
generally, constant step-size AMSGrad converges to a neighbourhood of the solution, attaining an
O(1/T + σ2) rate, which matches the rate of constant step-size SGD in the same setting (Schmidt &
Le Roux, 2013; Vaswani et al., 2019a). When training over-parameterized models, this result provides
√
some justiﬁcation for the faster (O(1/T ) vs. O(1/
T )) convergence of the AMSGrad variant typically
used in practice.

√

√

Adaptive step-size. Although AdaGrad converges at the same asymptotic rate for any step-size (up to
constants), it is unclear how to choose this step-size without manually trying different values. Similarly,
AMSGrad is sensitive to the step-size, converging only for a speciﬁc range in both theory and practice.
In Section 5, we experimentally show that even for simple, convex problems, the step-size has a big
impact on the empirical performance of AdaGrad and AMSGrad. To overcome this limitation, we
use recent methods (Vaswani et al., 2019a; Loizou et al., 2020) that automatically set the step-size
for SGD. These works use stochastic variants of the classical Armijo line-search (Armijo, 1966) or
the Polyak step-size (Polyak, 1963) in the interpolation setting. We combine these techniques with
adaptive gradient methods and show that a variant of stochastic line-search (SLS) enables AdaGrad to
adapt to the smoothness of the underlying function, resulting in faster empirical convergence, while
retaining its favourable convergence properties (Section 3). Similarly, AMSGrad with variants of SLS
and SPS can match the convergence rate of its constant step-size counterpart, but without knowledge
of the underlying smoothness properties (Section 4).

Experimental results. Finally, in Section 5, we benchmark our results against SGD variants with
SLS (Vaswani et al., 2019b), SPS (Loizou et al., 2020), tuned Adam and its recently proposed
variants (Luo et al., 2019; Liu et al., 2020). We demonstrate that the proposed techniques for setting
the step-size improve the empirical performance of adaptive gradient methods. These improvements
are consistent across tasks, ranging from binary classiﬁcation with a kernel mapping to multi-class
classiﬁcation using standard deep neural network architectures.

2

Under review as a conference paper at ICLR 2021

2 PROBLEM SETUP

(cid:80)n

We consider the unconstrained minimization of an objective f : Rd → R with a ﬁnite-sum structure,
f (w) = 1
i=1 fi(w). In supervised learning, n represents the number of training examples, and fi
n
is the loss function on training example i. Although we focus on the ﬁnite-sum setting, our results
can be easily generalized to the online optimization setting. The objective of our analysis is to better
understand the effect of the step-size and line-searches when interpolation is (almost) satisﬁed. This is
complicated by the fact that adaptive methods are still poorly understood; state-of-the-art analyses do
not show an improvement over gradient descent in the worst-case. To focus on the effect of step-sizes,
we make the simplifying assumptions described in this section.
We assume f and each fi are differentiable, convex, and lower-bounded by f ∗ and f ∗
i , respectively.
Furthermore, we assume that each function fi in the ﬁnite-sum is Li-smooth, implying that f is
Lmax-smooth, where Lmax = maxi Li. We also make the standard assumption that the iterates
remain bounded in a ball of radius D around a global minimizer, (cid:107)wk − w∗(cid:107) ≤ D for all wk (Ahn
et al., 2020). We remark that the bounded iterates assumption simpliﬁes the analysis but is not
essential, and similar to Reddi et al. (2018); Duchi et al. (2011); Levy et al. (2018), our theoretical
results can be extended to include a projection step. We include the formal deﬁnitions of these
properties (Nemirovski et al., 2009) in Appendix A.

The interpolation assumption means that the gradient of each fi in the ﬁnite-sum converges to zero
at an optimum. If the overall objective f is minimized at w∗, ∇f (w∗) = 0, then for all fi we have
∇fi(w∗) = 0. The interpolation condition can be exactly satisﬁed for many over-parameterized
machine learning models such as non-parametric kernel regression without regularization (Belkin
et al., 2019; Liang & Rakhlin, 2018) and over-parameterized deep neural networks (Zhang et al.,
2017). We measure the extent to which interpolation is violated by the disagreement between
the minimum overall function value f ∗ and the minimum value of each individual functions f ∗
i ,
σ2 := Ei[f ∗ − f ∗
i ] ∈ [0, ∞) (Loizou et al., 2020). The minimizer of f need not be unique for σ2 to
be uniquely deﬁned, as it only depends on the minimum function values. Interpolation is said to be
exactly satisﬁed if σ2 = 0, and we also study the setting when σ2 > 0.

For a preconditioner matrix Ak and a constant momentum parameter β ∈ [0, 1), the update for a
generic adaptive gradient method at iteration k can be expressed as:

wk+1 = wk − ηk A−1

k mk

; mk = βmk−1 + (1 − β)∇fik (wk)

(1)

Here, ∇fik (wk) is the stochastic gradient of a randomly chosen function fik , and ηk is the step-size.
Adaptive gradient methods typically differ in how their preconditioners are constructed and whether
or not they include the momentum term βmk−1 (see Table 1 for a list of common methods). Both

Table 1: Adaptive preconditioners (analyzed methods are bolded), with G0 = 0 and β1, β2 ∈ [0, 1). In
practice, a small (cid:15)I is added to ensure Ak (cid:31) 0. *: We use the PyTorch implementation in experiments
which includes bias correction.

Optimizer

Gk

(∇k := ∇fik (wk))

Ak

AdaGrad
RMSProp
Adam
AMSGrad*

(cid:62))

Gk−1 + diag(∇k∇k
(cid:62))
β2Gk−1 + (1 − β2) diag(∇k∇k
(cid:62)))/(1 − βk
(β2Gk−1 + (1 − β2) diag(∇k∇k
(cid:62)))/(1 − βk
(β2Gk−1 + (1 − β2) diag(∇k∇k

1/2
G
k
1/2
G
k
1/2
2 ) G
k
2 ) max{Ak−1, G

1/2
k }

β

0
0
β1
β1

RMSProp and Adam maintain an exponential moving average of past stochastic gradients, but as
Reddi et al. (2018) pointed out, unlike AdaGrad, the corresponding preconditioners do not guarantee
that Ak+1 (cid:23) Ak and the resulting per-dimension step-sizes do not go to zero. This can lead to
large ﬂuctuations in the effective step-size and prevent these methods from converging. To mitigate
this problem, they proposed AMSGrad, which ensures Ak+1 (cid:23) Ak and the convergence of iterates.
Consequently, our theoretical results focus on AdaGrad, AMSGrad and other adaptive gradient
methods that ensure this monotonicity. However, we also considered RMSProp and Adam in our
experimental evaluation.

Although our theory holds for both the full matrix and diagonal variants (where Ak is a diagonal
matrix) of these methods, we use only the latter in experiments for scalability. The diagonal variants

3

Under review as a conference paper at ICLR 2021

perform a per-dimension scaling of the gradient and avoid computing the full matrix inverse, so their
per-iteration cost is the same as SGD, although with an additional O(d) memory. For AMSGrad, we
assume that the corresponding preconditioners are well-behaved in the sense that their eigenvalues are
bounded in an interval [amin, amax]. This is a common assumption made in the analysis of adaptive
methods. Moreover, for diagonal preconditioners, such a boundedness property is easy to verify, and
it is also inexpensive to maintain the desired range by projection. Our main theoretical results for
AdaGrad (Section 3) and AMSGrad (Section 4) are summarized in Table 2.

Table 2: Results for smooth, convex functions.

Method

AdaGrad

Step-size

Constant
Conservative Lipschitz LS

Adapts
to smoothness
(cid:55)
(cid:51)

AMSGrad

Constant

AMSGrad w/o momentum Armijo SLS

AMSGrad

Conservative Armijo SPS

(cid:55)

(cid:51)

(cid:51)

3 ADAGRAD

Rate

Reference

√
O(1/T + σ/
√
O(1/T + σ/
O(1/T + σ2)
O(1/T + σ2)
O(1/T + σ2)

T ) Theorem 1
T ) Theorem 2

Theorem 3

Theorem 4

Theorem 5

T ), where ζ 2 = supw

For smooth, convex objectives, Levy et al. (2018) showed that AdaGrad converges at a rate
√
Ei[(cid:107)∇f (w) − ∇fi(w)(cid:107)2] is a uniform bound on the variance
O(1/T + ζ/
of the stochastic gradients. In the over-parameterized setting, we show that AdaGrad achieves the
O(1/T ) rate when interpolation is exactly satisﬁed and a slower convergence to the solution if
interpolation is violated.1 The proofs for this section are in Appendix C.
Theorem 1 (Constant step-size AdaGrad). Assuming (i) convexity and (ii) Lmax-smoothness of each
fi, and (iii) bounded iterates, AdaGrad with a constant step-size η and uniform averaging such that
¯wT = 1
T

k=1 wk, converges at a rate

(cid:80)T

E[f ( ¯wT ) − f ∗] ≤

α
T

+

√
√

ασ
T

, where α =

(cid:18) D2
η

1
2

(cid:19)2

+ 2η

dLmax.

When interpolation is exactly satisﬁed, a similar proof technique can be used to show that AdaGrad
incurs only O(1) regret in the online convex optimization setting (Theorem 6 in Appendix C.2). The
above theorem shows that AdaGrad is robust to the violation of interpolation and converges to the
minimizer at the desired rate for any reasonable step-size. Although this is a favourable property, the
best constant step-size depends on the problem, and as we demonstrate experimentally in Section 5,
the performance of AdaGrad depends on correctly tuning this step-size.

To overcome this limitation, we use a conservative Lipschitz line-search that sets the step-size on
the ﬂy, improving the empirical performance of AdaGrad while retaining its favourable convergence
guarantees. At each iteration, this line-search selects a step-size ηk that satisﬁes the property

fik (wk − ηk∇fik (wk)) ≤ fik (wk) − c ηk (cid:107)∇fik (wk)(cid:107)2 ,

and ηk ≤ ηk−1.

(2)

The resulting step-size is then used in the standard AdaGrad update in Eq. (1). To ﬁnd an acceptable
step, our results use a backtracking line-search, described in Appendix F. For simplicity, the theoretical
results assume access to the largest step-size that satisﬁes the above condition.2 Here, c is a hyper-
parameter determined theoretically and typically set to 1/2 in our experiments. The “conservative” part
of the line-search is the non-increasing constraint on the step-sizes, which is essential for convergence
to the minimizer when interpolation is violated. We refer to it as the Lipschitz line-search as it
is only used to estimate the local Lipschitz constant. Unlike the classical Armijo line-search for

1A similar result also appears in the course notes (Orabona, 2019).
2The difference between the exact and backtracking line-search is minimal, and the bounds are only changed

by a constant depending on the backtracking parameter.

4

Under review as a conference paper at ICLR 2021

preconditioned gradient descent, the line-search in Eq. (2) is in the gradient direction, even though
the update is in the preconditioned direction. The resulting step-size found is guaranteed to be in the
range [2(1−c)/Lmax, ηk−1] (Vaswani et al., 2019b) and allows us to prove the following theorem.
Theorem 2. Under the same assumptions as Theorem 1, AdaGrad with a conservative Lipschitz
line-search with c = 1/2, a step-size upper bound ηmax and uniform averaging converges at a rate

E[f ( ¯wT ) − f ∗] ≤

α
T

+

√
√

ασ
T

, where α =

(cid:18)

1
2

D2 max

(cid:26) 1

ηmax

(cid:27)

(cid:19)2

, Lmax

+ 2 ηmax

dLmax.

Intuitively, the Lipschitz line-search enables AdaGrad to take larger steps at iterates where the
underlying function is smoother. It retains the favourable convergence guarantees of constant step-
size AdaGrad, while improving its empirical performance (Section 5). Moreover, if interpolation is
exactly satisﬁed, we can obtain an O(1/T ) convergence without the conservative constraint ηk ≤ ηk−1
on the step-sizes (Appendix C.3).

4 AMSGRAD AND NON-DECREASING PRECONDITIONERS

In this section, we consider AMSGrad and, more generally, methods with non-decreasing precon-
ditioners satisfying Ak (cid:23) Ak−1. As our focus is on the behavior of the algorithm with respect to
the overall step-size, we make the simplifying assumption that the effect of the preconditioning
is bounded, meaning that the eigenvalues of Ak lie in the [amin, amax] range. This is a common
assumption made in the analyses of adaptive methods (Reddi et al., 2018; Alacaoglu et al., 2020) that
prove worst-case convergence rates matching those of SGD. For our theoretical results, we consider
the variant of AMSGrad without bias correction, as its effect is minimal after the ﬁrst few iterations.
The proofs for this section are in Appendix D and Appendix E.

The original analysis of AMSGrad (Reddi et al., 2018) uses a decreasing step-size and a decreasing
√
momentum parameter. It shows an O(1/
T ) convergence for AMSGrad in both the smooth and
non-smooth convex settings. Recently, Alacaoglu et al. (2020) showed that this analysis is loose
√
and that AMSGrad does not require a decreasing momentum parameter to obtain the O(1/
T ) rate.
However, in practice, AMSGrad is typically used with both a constant step-size and momentum
parameter. Next, we present the convergence result for this commonly-used variant of AMSGrad.
Theorem 3. Under the same assumptions as Theorem 1, and assuming (iv) non-decreasing precon-
ditioners (v) bounded eigenvalues in the [amin, amax] interval, where κ = amax/amin, AMSGrad with
β ∈ [0, 1), constant step-size η = 1−β
1+β

and uniform averaging converges at a rate,

amin
2Lmax

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 1 + β
1 − β

(cid:19)2 2LmaxD2dκ
T

+ σ2.

When σ = 0, we obtain a O(1/T ) convergence to the minimizer. However, when interpolation is only
approximately satisﬁed, we obtain convergence to a neighbourhood with its size depending on σ2. We
observe that the noise σ2 is not ampliﬁed because of the non-decreasing momentum (or step-size). A
similar distinction between the convergence of constant step-size Adam (or AMSGrad) vs. AdaGrad
has also been recently discussed in the non-convex setting (D´efossez et al., 2020). Unfortunately, the
ﬁnal bound is minimized by setting β1 = 0 and our theoretical analysis does not show an advantage
of using momentum. Note that this is a common drawback in the analyses of heavy-ball momentum
for non-quadratic functions in both the stochastic and deterministic settings (Ghadimi et al., 2015;
Reddi et al., 2018; Alacaoglu et al., 2020; Sebbouh et al., 2020).
Since AMSGrad is typically used for optimizing over-parameterized models, the violation σ2 is small,
even when interpolation is not exactly satisﬁed. Another reason that constant step-size AMSGrad
is practically useful is because of the use of large batch-sizes that result in a smaller effective
neighbourhood. To get some intuition about the effect of batch-size, note that if we use a batch-size
of b, the resulting neighbourhood depends on σ2
B is the
B(cid:107) (cid:107)∇fB(w∗)(cid:107)]. If
minimizer of a batch B of training examples. By convexity, σ2
b ∝ E(cid:107)∇fB(w∗)(cid:107). Since the examples in
we assume that the distance (cid:107)w∗ − x∗
nb (cid:107)∇fi(w∗)(cid:107),
each batch are sampled with replacement, using the bounds in (Lohr, 2009), σ2
showing that the effective neighbourhood shrinks as the batch-size becomes larger, becoming zero for

b := EB;|B|=b [fB(w∗) − fB(x∗
b ≤ E[(cid:107)w∗ − x∗

B(cid:107) is bounded, σ2

B)] where w∗

b ∝ n−b

5

Under review as a conference paper at ICLR 2021

the full-batch variant. With over-parameterization and large batch-sizes, the effective neighbourhood
is small enough for machine learning tasks that do not require exact convergence to the solution.

The constant step-size required for the above result depends on Lmax, which is typically unknown.
Furthermore, using a global bound on Lmax usually results in slower convergence since the local
Lipschitz constant can vary considerably during the optimization. To overcome these issues, we use a
stochastic variant of the Armijo line-search. Unlike the Lipschitz line-search whose sole purpose is
to estimate the Lipschitz constant, the Armijo line-search selects a suitable step-size in the precon-
ditioned gradient direction, and as we show in Section 5, it results in better empirical performance.
Similar to the constant step-size, when interpolation is violated, we only obtain convergence to a
neighbourhood of the solution. The stochastic Armijo line-search returns the largest step-size ηk
satisfying the following conditions at iteration k,

fik (wk − ηkA−1

k ∇fik (wk)) ≤ fik (wk) − c ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

,

and ηk ≤ ηmax.

(3)

The step-size is artiﬁcially upper-bounded by ηmax (typically chosen to be a large value). The line-
search guarantees descent on the current function fik and that ηk lies in the [2amin (1−c)/Lmax, ηmax]
range. In the next theorem, we ﬁrst consider the variant of AMSGrad without momentum (β = 0)
and show that using the Armijo line-search retains the O(1/T ) convergence rate without the need to
know the Lipschitz constant.
Theorem 4. Under the same assumptions as Theorem 1, AMSGrad with zero momentum, Armijo
line-search with c = 3/4, a step-size upper bound ηmax and uniform averaging converges at a rate,

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 3D2d · amax
2T

+ 3ηmaxσ2

(cid:19)

max

(cid:26) 1

ηmax

,

2Lmax
amin

(cid:27)

.

Comparing this rate with that of using constant step-size (Theorem 3), we observe that the Armijo
line-search results in a worse constant in the convergence rate and a larger neighbourhood. These
dependencies can be improved by considering a conservative version of the Armijo line-search.
However, we experimentally show that the proposed line-search drastically improves the empirical
performance of AMSGrad. We show that a similar bound also holds for AdaGrad (see Theorem 7
in Appendix C). AdaGrad with an Armijo line-search converges to a neighbourhood in the absence
of interpolation (unlike the results in 3). Moreover, the above bound depends on amin which can
be O((cid:15)) in the worst-case, resulting in an unsatisfactory worst-case rate of O(1/(cid:15)T ) even in the
interpolation setting. However, like AMSGrad, AdaGrad with Armijo line-search has excellent
empirical performance, implying the need for a different theoretical assumption in the future.

Before considering techniques to set the step-size for AMSGrad including momentum, we present
the details of the stochastic Polyak step-size (SPS) Loizou et al. (2020); Berrada et al. (2019) and
Armijo SPS, our modiﬁcation to the adaptive setting. These variants set the step-size as:

SPS: ηk = min

(cid:40)

∗
fik (wk) − fik
c (cid:107)∇fik (wk)(cid:107)2 , ηmax

(cid:41)

, Armijo SPS: ηk = min

(cid:40)

∗
fik (wk) − fik
c (cid:107)∇fik (wk)(cid:107)2

A−1
k

(cid:41)

, ηmax

.

∗ is the minimum value for the function fik .The advantage of SPS over a line-search is that it
Here, fik
does not require a potentially expensive backtracking procedure to set the step-size. Moreover, it can
be shown that this step-size is always larger than the one returned by line-search, which can lead to
faster convergence. However, SPS requires knowledge of f ∗
i for each function in the ﬁnite-sum. This
value is difﬁcult to obtain for general functions but is readily available in the interpolation setting for
many machine learning applications. Common loss functions are lower-bounded by zero, and the
interpolation setting ensures that these lower-bounds are tight. Consequently, using SPS with f ∗
i = 0
has been shown to yield good performance for over-parameterized problems (Loizou et al., 2020;
Berrada et al., 2019). In Appendix D, we show that the Armijo line-search used for the previous
results can be replaced by Armijo SPS and result in similar convergence rates.

For AMSGrad with momentum, we propose to use a conservative variant of Armijo SPS that sets
ηmax = ηk−1 at iteration k ensuring that ηk ≤ ηk−1. This is because using a potentially increasing
step-size sequence along with momentum can make the optimization unstable and result in divergence.
Using this step-size, we prove the following result.
Theorem 5. Under the same assumptions of Theorem 1 and assuming (iv) non-decreasing precon-
ditioners (v) bounded eigenvalues in the [amin, amax] interval with κ = amax/amin, AMSGrad with

6

Under review as a conference paper at ICLR 2021

β ∈ [0, 1), conservative Armijo SPS with c = 1+β/1−β and uniform averaging converges at a rate,

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 1 + β
1 − β

(cid:19)2 2LmaxD2dκ
T

+ σ2.

The above result exactly matches the convergence rate in Theorem 3 but does not require knowledge of
the smoothness constant to set the step-size. Moreover, the conservative step-size enables convergence
without requiring an artiﬁcial upper-bound ηmax as in Theorem 8. We note that a similar convergence
rate can be obtained when using a conservative variant of Armijo SLS ( Appendix E.2), although our
theoretical techniques only allow for a restricted range of β.

When Ak = Id, the AMSGrad update is equivalent to the update for SGD with heavy-ball momen-
tum (Sebbouh et al., 2020). By setting Ak = Id in the above result, we recover an O(1/T + σ2)
rate for SGD (using SPS to set the step-size) with heavy-ball momentum. In the smooth, convex
setting, our rate matches that of (Sebbouh et al., 2020); however, unlike their result, we do not require
knowledge of the Lipschitz constant. This result also provides theoretical justiﬁcation for the heuristic
used for incorporating heavy-ball momentum for SLS in (Vaswani et al., 2019b).

For a general preconditioner, the AMSGrad update in Eq. (1) is not equivalent to heavy-ball mo-
mentum. With a constant momentum parameter γ ∈ [0, 1), the general heavy-ball update (Loizou &
Richt´arik, 2017) is given as wk+1 = wk −αk A−1
k ∇fik (wk)+γ (wk − wk−1) (refer to Appendix E.1
for a relation between the two updates). Unlike this update, AMSGrad also preconditions the momen-
tum direction (wk − wk−1). If we consider the zero-momentum variant of adaptive gradient methods
as preconditioned gradient descent, the above update is a more natural way to incorporate momentum.
We explore this alternate method and prove the same O(1/T + σ2) convergence rate for constant
step-size, conservative Armijo SPS and Armijo SLS techniques in Appendix E.3. In the next section,
we use the above techniques for training large over-parameterized deep networks.

5 EXPERIMENTAL EVALUATION

(a) AdaGrad

(b) AMSGrad

Figure 1: Synthetic experiments showing the impact of step-size on the performance of AdaGrad,
AMSGrad with varying step-sizes, including the default in PyTorch, and the SLS variants.

Synthetic experiment: We ﬁrst present an experiment to show that AdaGrad and AMSGrad
with constant step-size are not robust even for simple, convex problems. We use their PyTorch
implementations (Paszke et al., 2019) on a binary classiﬁcation task with logistic regression. Following
the protocol of Meng et al. (2020), we generate a linearly-separable dataset with n = 103 examples
(ensuring interpolation is satisﬁed) and d = 20 features with varying margins. For AdaGrad and
AMSGrad with a batch-size of 100, we show the training loss for a grid of step-sizes in the [103, 10−3]
range and also plot their default (in PyTorch) variants. For AdaGrad, we compare against the proposed
Lipschitz line-search and Armijo SLS variants. As is suggested by the theory, for each of these variants,
we set the value of c = 1/2. For AMSGrad, we compare against the variant employing the Armijo
SLS with c = 1/2.3 and use the default (in PyTorch) momentum parameter of β = 0.9. In Fig. 1, we
observe a large variance across step-sizes and poor performance of the default step-size. The best
performing variant of AdaGrad/AMSGrad has a step-size of order 102. The line-search variants have
good performance across margins, often better than the best-performing constant step-size.

3This corresponds to the largest allowable step-size in Theorem 4 without momentum. Unfortunately, the

values of c suggested by the analysis incorporating momentum Theorem 5 are too conservative.

7

050100150200Epoch102101100101Train loss (log)Margin:0.01050100150200Epoch102101100101Margin:0.05AdagradDefault AdagradAdagrad + Lipschitz LSAdagrad + Armijo LS050100150200Epoch102101100101Train loss (log)Margin:0.01050100150200Epoch107105103101101Margin:0.05AmsgradDefault AmsgradAmsgrad + SLSUnder review as a conference paper at ICLR 2021

Figure 2: Comparing optimizers for multi-class classiﬁcation with deep networks. Training loss (top)
and validation accuracy (bottom) for CIFAR-10, CIFAR-100 and Tiny ImageNet.

Real experiments: Following the protocol in (Luo et al., 2019; Vaswani et al., 2019b; Loizou et al.,
2020), we consider training standard neural network architectures for multi-class classiﬁcation on
CIFAR-10, CIFAR-100 and variants of the ImageNet datasets. For each of these experiments, we use
a batch-size of 128 and compare against Adam with the best constant step-size found by grid-search.
We also include recent improved variants of Adam; RAdam (Liu et al., 2020) and AdaBound (Luo
et al., 2019). To see the effect of preconditioning, we compare against SGD with SLS (Vaswani et al.,
2019a) and SPS (Loizou et al., 2020). We ﬁnd that SGD with SLS is more stable and has consistently
better test performance than SPS, and hence we only show results for SLS. We also compared against
tuned constant step-size SGD and similar to (Vaswani et al., 2019a), we observe that it is consistently
outperformed by SGD with SLS.

For the proposed methods, we consider the combinations with theoretical guarantees in the convex
setting, speciﬁcally AdaGrad and AMSGrad with the Armijo SLS. For AdaGrad, we only show
Armijo SLS since it consistently outperforms the Lipschitz line-search. For all variants with Armijo
SLS, we use c = 0.5 for all convex experiments (suggested by Theorem 4 and Vaswani et al.
(2019a)). Since we do not have a theoretical analysis for non-convex problems, we follow the protocol
in Vaswani et al. (2019a) and set c = 0.1 for all the non-convex experiments. Throughout, we set
β = 0.9 for AMSGrad. We also compare to the AMSGrad variant with heavy-ball (HB) momentum
(with γ = 0.25 found by grid-search). We refer to Appendix F for a detailed discussion about the
practical considerations and pseudocodes for the SLS/SPS variants.

We show a subset of results for CIFAR-10, CIFAR-100 and Tiny ImageNet and defer the rest
to Appendix G. From Fig. 2 we make the following observations, (i) in terms of generalization,
AdaGrad and AMSGrad with Armijo SLS have consistently the best performance, while SGD with
SLS is often competitive. (ii) the AdaGrad and AMSGrad variants not only converge faster than
Adam and Radam but also with considerably better test performance. AdaBound has comparable
convergence in terms of training loss, but does not generalize as well. (iii) AMSGrad momentum
is consistently better than the heavy-ball (HB) variant. Moreover, we observed that HB momentum
was quite sensitive to the setting of γ, whereas AMSGrad is robust to β. In Appendix G, we include
ablation results for AMSGrad with Armijo SLS but without momentum, and conclude that momentum
does indeed improve the performance. In Appendix G, we plot the wall-clock time for the SLS variants
and verify that the performance gains justify the increase in wall-clock time per epoch. In the appendix,
we show the variation of step-size across epochs, observing a warm-up phase where the step-size
increases followed by a constant or decreasing step-size (Goyal et al., 2017).

8

050100150200Epoch104103102101100Train loss (log)CIFAR10 - ResNet3450100150200Epoch103102101100CIFAR100 - DenseNet121050100150200Epoch103102101100CIFAR100 - ResNet34050100150200Epoch103102101100101Tiny ImageNet - ResNet18050100150200Epoch0.860.880.900.920.94Validation accuracyCIFAR10 - ResNet3450100150200Epoch0.660.680.700.720.740.76CIFAR100 - DenseNet121050100150200Epoch0.660.680.700.720.740.76CIFAR100 - ResNet3450100150200Epoch0.340.350.360.370.380.390.40Tiny ImageNet - ResNet18Amsgrad + SLSAmsgrad +  SLS + HBAdaboundRadamAdamSLSAdagrad +  SLSUnder review as a conference paper at ICLR 2021

In Appendix G, we also consider binary classiﬁcation with RBF kernels for datasets from LIB-
SVM (Chang & Lin, 2011) and study the effect of over-parameterization for deep matrix factoriza-
tion (Rolinek & Martius, 2018; Vaswani et al., 2019b). We show that the same trends hold across
different datasets, deep models, deep matrix factorization, and binary classiﬁcation using kernels.

Our results indicate that simply setting the correct step-size on the ﬂy can lead to substantial empirical
gains, often more than those obtained by designing a different preconditioner. Furthermore, we
see that with an appropriate step-size adaptation, adaptive gradient methods can generalize better
than SGD. By disentangling the effect of the step-size from the preconditioner, our results show
that AdaGrad has good empirical performance, contradicting common knowledge. Moreover, our
techniques are orthogonal to designing better preconditioners and can be used with other adaptive
gradient or even second-order methods.

6 DISCUSSION

When training over-parameterized models in the interpolation setting, we showed that for smooth,
convex functions, constant step-size variants of both AdaGrad and AMSGrad are guaranteed to
converge to the minimizer at O(1/T ) rates. We proposed to use stochastic line-search techniques
to help these methods adapt to the function’s local smoothness, alleviating the need to tune their
step-size and resulting in consistent empirical improvements across tasks. Although adaptive gradient
methods outperform SGD in practice, their convergence rates are worse than constant step-size SGD
and we hope to address this discrepancy in the future.

REFERENCES

Kwangjun Ahn, Chulhee Yun, and Suvrit Sra. Sgd with shufﬂing: optimal rates without component
convexity and large epoch requirements. Advances in Neural Information Processing Systems, 33,
2020.

Ahmet Alacaoglu, Yura Malitsky, Panayotis Mertikopoulos, and Volkan Cevher. A new regret analysis

for adam-type algorithms. arXiv preprint arXiv:2003.09729, 2020.

Larry Armijo. Minimization of functions having lipschitz continuous ﬁrst partial derivatives. Paciﬁc

Journal of mathematics, 16(1):1–3, 1966.

Mikhail Belkin, Alexander Rakhlin, and Alexandre B. Tsybakov. Does data interpolation contradict
statistical optimality? In The 22nd International Conference on Artiﬁcial Intelligence and Statistics,
AISTATS, 2019.

Leonard Berrada, Andrew Zisserman, and M. Pawan Kumar. Training neural networks for and by

interpolation. arXiv preprint:1906.05661, 2019.

Nicol`o Cesa-Bianchi, Alex Conconi, and Claudio Gentile. On the generalization ability of on-line

learning algorithms. IEEE Transactions on Information Theory, 50(9):2050–2057, 2004.

Volkan Cevher and Bang Cˆong V˜u. On the linear convergence of the stochastic gradient method with

constant step-size. Optimization Letters, 13(5):1177–1187, 2019.

Chih-Chung Chang and Chih-Jen Lin. LIBSVM: A library for support vector machines. ACM
Transactions on Intelligent Systems and Technology, 2(3):1–27, 2011. Software available at
http://www.csie.ntu.edu.tw/˜cjlin/libsvm.

Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the convergence of a class of Adam-
In 7th International Conference on Learning

type algorithms for non-convex optimization.
Representations, ICLR, 2019.

Alexandre D´efossez, L´eon Bottou, Francis Bach, and Nicolas Usunier. On the convergence of Adam

and AdaGrad. arXiv preprint:2003.02395, 2020.

John C. Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning
and stochastic optimization. The Journal of Machine Learning Research, 12:2121–2159, 2011.

9

Under review as a conference paper at ICLR 2021

Euhanna Ghadimi, Hamid Reza Feyzmahdavian, and Mikael Johansson. Global convergence of
the heavy-ball method for convex optimization. In 2015 European control conference (ECC), pp.
310–315. IEEE, 2015.

Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep learning. Adaptive computation and

machine learning. MIT press, 2016. URL http://www.deeplearningbook.org/.

Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola,
Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: training
imagenet in 1 hour. arXiv preprint:1706.02677, 2017.

Elad Hazan. Introduction to online convex optimization. Foundations and Trends in Optimization, 2

(3-4):157–325, 2016.

Elad Hazan and Satyen Kale. Beyond the regret minimization barrier: optimal algorithms for
stochastic strongly-convex optimization. The Journal of Machine Learning Research, 15(1):
2489–2512, 2014.

Prateek Jain, Sham M. Kakade, Rahul Kidambi, Praneeth Netrapalli, and Aaron Sidford. Accelerating
stochastic gradient descent for least squares regression. In Conference On Learning Theory, COLT,
2018.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In 3rd International

Conference on Learning Representations, ICLR, 2015.

Kﬁr Y. Levy, Alp Yurtsever, and Volkan Cevher. Online adaptive methods, universality and accelera-

tion. In Advances in Neural Information Processing Systems, NeurIPS, 2018.

Xiaoyu Li and Francesco Orabona. On the convergence of stochastic gradient descent with adaptive
stepsizes. In The 22nd International Conference on Artiﬁcial Intelligence and Statistics, AISTATS,
2019.

Tengyuan Liang and Alexander Rakhlin. Just interpolate: Kernel “ridgeless” regression can generalize.

arXiv preprint:1808.00387, 2018.

Chaoyue Liu and Mikhail Belkin. Accelerating SGD with momentum for over-parameterized learning.

In 8th International Conference on Learning Representations, ICLR, 2020.

Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei
Han. On the variance of the adaptive learning rate and beyond. In 8th International Conference on
Learning Representations, ICLR, 2020.

Sharon L Lohr. Sampling: design and analysis. Nelson Education, 2009.

Nicolas Loizou and Peter Richt´arik. Linearly convergent stochastic heavy ball method for minimizing

generalization error. arXiv preprint:1710.10737, 2017.

Nicolas Loizou, Sharan Vaswani, Issam Laradji, and Simon Lacoste-Julien. Stochastic Polyak
step-size for SGD: An adaptive learning rate for fast convergence. arXiv preprint:2002.10542,
2020.

Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts.

In
5th International Conference on Learning Representations, ICLR. OpenReview.net, 2017. URL
https://openreview.net/forum?id=Skq89Scxx.

Liangchen Luo, Yuanhao Xiong, Yan Liu, and Xu Sun. Adaptive gradient methods with dynamic
bound of learning rate. In 7th International Conference on Learning Representations, ICLR, 2019.

Siyuan Ma, Raef Bassily, and Mikhail Belkin. The power of interpolation: Understanding the effec-
tiveness of SGD in modern over-parametrized learning. In Proceedings of the 35th International
Conference on Machine Learning, ICML, 2018.

Si Yi Meng, Sharan Vaswani, Issam Laradji, Mark Schmidt, and Simon Lacoste-Julien. Fast and furi-
ous convergence: Stochastic second order methods under interpolation. In The 23nd International
Conference on Artiﬁcial Intelligence and Statistics, AISTATS, 2020.

10

Under review as a conference paper at ICLR 2021

Eric Moulines and Francis R. Bach. Non-asymptotic analysis of stochastic approximation algorithms
for machine learning. In Advances in Neural Information Processing Systems, NeurIPS, 2011.

Mahesh Chandra Mukkamala and Matthias Hein. Variants of RMSProp and AdaGrad with logarithmic
regret bounds. In Proceedings of the 34th International Conference on Machine Learning, ICML,
2017.

Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic
approximation approach to stochastic programming. SIAM Journal on Optimization, 19(4):1574–
1609, 2009.

Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213,

2019.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward
Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep
learning library. In Advances in Neural Information Processing Systems, NeurIPS, 2019.

Boris T. Polyak. Gradient methods for minimizing functionals. Zhurnal Vychislitel’noi Matematiki i

Matematicheskoi Fiziki, 3(4):643–653, 1963.

Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of Adam and beyond. In 6th

International Conference on Learning Representations, ICLR, 2018.

Michal Rolinek and Georg Martius. L4: practical loss-based stepsize adaptation for deep learning. In

Advances in Neural Information Processing Systems, NeurIPS, 2018.

Mark Schmidt and Nicolas Le Roux. Fast convergence of stochastic gradient descent under a strong

growth condition. arXiv preprint:1308.6370, 2013.

Othmane Sebbouh, Robert M Gower, and Aaron Defazio. On the convergence of the stochastic heavy

ball method. arXiv preprint arXiv:2006.07867, 2020.

Matthew Staib, Sashank J. Reddi, Satyen Kale, Sanjiv Kumar, and Suvrit Sra. Escaping saddle points
with adaptive gradient methods. In Proceedings of the 36th International Conference on Machine
Learning, ICML, 2019.

Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-RMSProp: Divide the gradient by a running
average of its recent magnitude. COURSERA: Neural networks for machine learning, 2012.

Sharan Vaswani, Francis Bach, and Mark Schmidt. Fast and faster convergence of SGD for over-
parameterized models and an accelerated perceptron. In The 22nd International Conference on
Artiﬁcial Intelligence and Statistics, AISTATS, 2019a.

Sharan Vaswani, Aaron Mishkin, Issam Laradji, Mark Schmidt, Gauthier Gidel, and Simon Lacoste-
Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. In Advances
in Neural Information Processing Systems, NeurIPS, 2019b.

Guanghui Wang, Shiyin Lu, Quan Cheng, Weiwei Tu, and Lijun Zhang. SAdam: A variant of Adam
for strongly convex functions. In 8th International Conference on Learning Representations, ICLR,
2020.

Rachel Ward, Xiaoxia Wu, and Leon Bottou. AdaGrad stepsizes: Sharp convergence over nonconvex
In Proceedings of the 36th International Conference on

landscapes, from any initialization.
Machine Learning, ICML, 2019.

Xiaoxia Wu, Simon S. Du, and Rachel Ward. Global convergence of adaptive gradient methods for

an over-parameterized neural network. arXiv preprint:1902.07111, 2019.

Yuege Xie, Xiaoxia Wu, and Rachel Ward. Linear convergence of adaptive stochastic gradient descent.
In Silvia Chiappa and Roberto Calandra (eds.), The 23rd International Conference on Artiﬁcial
Intelligence and Statistics, AISTATS, volume 108 of Proceedings of Machine Learning Research,
pp. 1475–1485. PMLR, 2020.

11

Under review as a conference paper at ICLR 2021

Matthew D. Zeiler. ADADELTA: an adaptive learning rate method. arXiv preprint:1212.5701, 2012.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding
deep learning requires rethinking generalization. In 5th International Conference on Learning
Representations, ICLR, 2017.

Lijun Zhang and Zhi-Hua Zhou. Stochastic approximation of smooth and strongly convex functions:

Beyond the O(1/T ) convergence rate. In Conference on Learning Theory, COLT, 2019.

Dongruo Zhou, Yiqi Tang, Ziyan Yang, Yuan Cao, and Quanquan Gu. On the convergence of adaptive

gradient methods for nonconvex optimization. arXiv preprint:1808.05671, 2018.

12

Under review as a conference paper at ICLR 2021

Supplementary material

ORGANIZATION OF THE APPENDIX

A Setup and assumptions

B Line-search and Polyak step-sizes
C Proofs for AdaGrad

Step-size

Constant
Conservative Lipschitz LS
Non-conservative LS (with interpolation)

Rate

√
O(1/T + σ/
√
O(1/T + σ/
O(1/T )

T )
T )

Reference

Theorem 1
Theorem 2
Theorem 7

D Proofs for AMSGrad and non-decreasing preconditioners without momentum

Constant
Armijo LS

E AMSGrad with momentum

Constant
Conservative Armijo LS
Conservative Armijo SPS

O(1/T + σ2)
O(1/T + σ2)

Theorem 8
Theorem 4

O(1/T + σ2)
O(1/T + σ2)
O(1/T + σ2)

Theorem 3
Theorem 10
Theorem 5

Proofs for AMSGrad with heavy ball momentum

Constant
Conservative Armijo LS
Conservative Armijo SPS

F Experimental details

G Additional experimental results

O(1/T + σ2)
O(1/T + σ2)
O(1/T + σ2)

Theorem 11
Theorem 13
Theorem 12

13

Under review as a conference paper at ICLR 2021

Table 3: Summary of notation

Concept

Symbol

Concept

Symbol

k, T
Iteration counter, maximum
wk, w∗
Iterates, minimum
ηk
Step-size
f (w), f ∗
Function value, minimum
Stoch. function value, minimum fi(w), f ∗
i

General preconditioner Ak
Preconditioner bounds
Maximum smoothness
Dimensionality
Diameter bound
Variance

[amin, amax]
Lmax
d
D
σ2 = Ei[fi(w∗) − f ∗
i ]

A SETUP AND ASSUMPTIONS

We restate the main notation in Table 3. We now restate the main assumptions required for our
theoretical results
We assume our objective f : Rd → R has a ﬁnite-sum structure,

f (w) =

1
n

n
(cid:88)

i=1

fi(w),

(4)

and analyze the following update, with ik selected uniformly at random,

wk+1 = wk − ηk A−1

k mk

; mk = βmk−1 + (1 − β)∇fik (wk)

(Update rule)

where ηk is either a pre-speciﬁed constant or selected on the ﬂy. We consider AdaGrad and AMSGrad
and use the fact that the preconditioners are non-decreasing i.e. Ak (cid:23) Ak−1. For AdaGrad, β = 0.
For AMSGrad, we further assume that the preconditioners remain bounded with eigenvalues in the
range [amin, amax],

aminI (cid:22) Ak (cid:22) amaxI.

(Bounded preconditioner)

For all algorithms, we assume that the iterates do not diverge and remain in a ball of radius D, as
is standard in the literature on online learning (Duchi et al., 2011; Levy et al., 2018) and adaptive
gradient methods (Reddi et al., 2018),

(cid:107)wk − w∗(cid:107) ≤ D.

(Bounded iterates)

Our main assumptions are that each individual function fi is convex, differentiable, has a ﬁnite
minimum f ∗

i , and is Li-smooth, meaning that for all v and w,
fi(v) ≥ fi(w) − (cid:104)∇fi(w), w − v(cid:105),

(Individual Convexity)

fi(v) ≤ fi(w) + (cid:104)∇fi(w), v − w(cid:105) +

Li
2
which also implies that f is convex and Lmax-smooth, where Lmax is the maximum smoothness
constant of the individual functions. A consequence of smoothness is the following bound on the
norm of the gradient stochastic gradients,

(Individual Smoothness)

(cid:107)v − w(cid:107)2 ,

(cid:107)∇fi(w)(cid:107)2 ≤ 2Lmax(fi(w) − f ∗
To characterize interpolation, we deﬁne the expected difference between the minimum of f , f (w∗),
and the minimum of the individual functions f ∗
i ,

i ).

σ2 = E
i

[fi(w∗) − f ∗

i ] < ∞.

(Noise)

When interpolation is exactly satisﬁed, every data point can be ﬁt exactly, such that f ∗
f (w∗) = 0, we have σ2 = 0.

i = 0 and

14

Under review as a conference paper at ICLR 2021

B LINE-SEARCH AND POLYAK STEP-SIZES

We now give the main guarantees on the step-sizes returned by the line-search. In practice, we use a
backtracking line-search to ﬁnd a step-size that satisﬁes the constraints, described in Algorithm 1
(Appendix F). For simplicity of presentation, here we assume the line-search returns the largest
step-size that satisﬁes the constraints.

When interpolation is not exactly satisﬁed, the procedures need to be equipped with an additional
safety mechanism; either by capping the maximum step-size by some ηmax or by ensuring non-
increasing step-sizes, ηk ≤ ηk−1. In this case, ηmax ensures that a bad iteration of the line-search
procedure does not result in divergence. When interpolation is satisﬁed, those conditions can be
dropped (e.g., setting ηmax → ∞) and the rate does not depend on ηmax. The line-searches depend
on a parameter c ∈ (0, 1) that controls how much decrease is necessary to accept a step (larger c
means more decrease is demanded).

Assuming the Lipschitz and Armijo line-searches select the largest η such that

fi(w − η∇fi(w)) ≤ fi(w) − cη (cid:107)∇fi(w)(cid:107)2 ,
fi(w − ηA−1∇fi(w)) ≤ fi(w) − cη (cid:107)∇fi(w)(cid:107)2

A−1 ,

η ≤ ηmax,

η ≤ ηmax,

(Lipschitz line-search)

(Armijo line-search)

the following lemma holds.

Lemma 1 (Line-search). If fi is Li-smooth, the Lipschitz and Armijo lines-searches ensure

η (cid:107)∇fi(w)(cid:107)2 ≤

η (cid:107)∇fi(w)(cid:107)2

A−1 ≤

1
c
1
c

(fi(w) − f ∗

i ),

and

(cid:26)

min

ηmax,

2 (1 − c)
Li

(fi(w) − f ∗

i ),

(cid:26)

and min

ηmax,

2 λmin(A) (1 − c)
Li

(cid:27)

(cid:27)

≤ η ≤ ηmax,

≤ η ≤ ηmax.

We do not include the backtracking line-search parameters in the analysis for simplicity, as the same
bounds hold, up to some constant. With a backtracking line-search, we start with a large enough
candidate step-size and multiply it by some constant γ < 1 until the Lipschitz or Armijo line-search
condition is satisﬁed. If η(cid:48) was a proposal step-size that did not satisfy the constraint, but γη(cid:48) does,
the maximum step-size η that satisﬁes the constraint must be in the range γη(cid:48) ≤ η < η(cid:48).

Proof of Lemma 1. Recall that if fi is Li-smooth, then for an arbitrary direction d,

fi(w − d) ≤ fi(w) − (cid:104)∇fi(w), d(cid:105) +

Li
2

(cid:107)d(cid:107)2 .

For the Lipschitz line-search, d = η∇fi(w). The smoothness and the line-search condition are then
fi(w − η∇fi(w)) − fi(w) ≤ (cid:0) Li
fi(w − η∇fi(w)) − fi(w) ≤ −cη (cid:107)∇fi(w)(cid:107)2 .

2 η2 − η(cid:1) (cid:107)∇fi(w)(cid:107)2 ,

Smoothness:

Line-search:

As illustrated in Fig. 3, the line-search condition
is looser than smoothness if

fi(w)
•

2 η2 − η(cid:1) (cid:107)∇fi(w)(cid:107)2 ≤ −cη (cid:107)∇fi(w)(cid:107)2 .
(cid:0) Li
The inequality is satisﬁed for any η ∈ [a, b],
where a, b are values of η that satisfy the equa-
tion with equality, a = 0, b = 2(1−c)/Li, and the
line-search condition holds for η ≤ 2(1−c)/Li.

Smoothness:
fi(w) + ( Li

2 η2 − η)(cid:107)∇fi(w)(cid:107)2

Line search:
fi(w) − cη(cid:107)∇fi(w)(cid:107)2

η = 0

η = 2(1−c)

Li

Figure 3: Sketch of the line-search inequalities.

As the line-search selects the largest feasible step-size, η ≥ 2(1−c)/Li. If the step-size is capped at
ηmax, we have η ≥ min{ηmax, 2(1−c)/Li}, and the proof for the Lipschitz line-search is complete.
The proof for the Armijo line-search is identical except for the smoothness property, which is modiﬁed

15

Under review as a conference paper at ICLR 2021

to use the (cid:107)·(cid:107)A-norm for the direction d = ηA−1∇fi(w);

fi(w − ηA−1∇fi(w)) ≤ fi(w) − η(cid:104)∇fi(w), A−1∇fi(w)(cid:105) +

(cid:13)A−1∇fi(w)(cid:13)
2
(cid:13)

η2 (cid:13)

Li
2
η2 (cid:107)∇fi(w)(cid:107)2

A−1 ,

,

≤ fi(w) − η (cid:107)∇fi(w)(cid:107)2

(cid:18)

= fi(w)+

Li
2λmin(A)

A−1 +
(cid:19)

η2 − η

Li
2λmin(A)

(cid:107)∇fi(w)(cid:107)2

A−1 ,

where the second inequality comes from (cid:107)A−1∇fi(w)(cid:107)2 ≤

1

λmin(A) (cid:107)∇fi(w)(cid:107)2

A−1.

Similarly, the stochastic Polyak step-sizes (SPS) for fi at w are deﬁned as

SPS:

η = min

(cid:40)

(cid:41)

fi(w) − f ∗
i
c (cid:107)∇fi(w)(cid:107)2 , ηmax

, Armijo SPS:

η = min

(cid:40)

fi(w) − f ∗
i
c (cid:107)∇fi(w)(cid:107)2

A−1

(cid:41)

, ηmax

,

where the parameter c > 0 controls the scaling of the step (larger c means smaller steps).

Lemma 2 (SPS guarantees). If fi is Li-smooth, SPS and Armijo SPS ensure that

SPS:

Armijo SPS:

η (cid:107)∇fi(w)(cid:107)2 ≤ 1

c (fi(w) − f ∗

i ),

η (cid:107)∇fi(w)(cid:107)2

A−1 ≤ 1

c (fi(w) − f ∗

i ), min

(cid:110)

ηmax,

1
min
2cLi
(cid:110)
ηmax, λmin(A)

2cLi

(cid:111)

(cid:111)

≤ η ≤ ηmax,

≤ η ≤ ηmax

Proof of Lemma 2. The ﬁrst guarantee follows directly from the deﬁnition of the step-size. For SPS,

η (cid:107)∇fi(w)(cid:107)2 = min

= min

(cid:40)

fi(w) − f ∗
i
c (cid:107)∇fi(w)(cid:107)2 , ηmax
(cid:26) fi(w) − f ∗
c

i

(cid:41)

(cid:107)∇fi(w)(cid:107)2 ,

, ηmax (cid:107)∇fi(w)(cid:107)2

(cid:27)

≤

1
c

(fi(w) − f (cid:63)

i ).

The same inequalities hold for Armijo SPS with (cid:107)∇fi(w)(cid:107)2
A−1 . To lower-bound the step-size, we
i ≥ 1
use the Li-smoothness of fi, which implies fi(w) − f ∗
2Li
(cid:107)∇fi(w)(cid:107)2
c (cid:107)∇fi(w)(cid:107)2 =
A−1 ≤

For Armijo SPS, we additionally use (cid:107)∇fi(w)(cid:107)2

fi(w) − f ∗
i
c (cid:107)∇fi(w)(cid:107)2 ≥

(cid:107)∇fi(w)(cid:107)2. For SPS,

λmin(A) (cid:107)∇fi(w)(cid:107)2,

1
2cLi

1
2Li

1

.

fi(w) − f ∗
i
c (cid:107)∇fi(w)(cid:107)2

A−1

≥

1
2Li
1

(cid:107)∇fi(w)(cid:107)2
λmin(A) (cid:107)∇fi(w)(cid:107)2 =

c

λmin(A)
2cLi

.

16

Under review as a conference paper at ICLR 2021

C PROOFS FOR ADAGRAD

We now move to the proof of the convergence of AdaGrad in the smooth setting with a constant
step-size (Theorem 1) and the conservative Lipschitz line-search (Theorem 2). We ﬁrst give a rate
for an arbitrary step-size ηk in the range [ηmin, ηmax], and derive the rates of Theorems 1 and 2 by
specializing the range to a constant step-size or line-search.

Proposition 1 (AdaGrad with non-increasing step-sizes). Assuming (i) convexity and (ii) Lmax-
smoothness of each fi, and (iii) bounded iterates, AdaGrad with non-increasing (ηk ≤ ηk−1),
bounded step-sizes (ηk ∈ [ηmin, ηmax]), and uniform averaging ¯wT = 1
k=1wk, converges at a
T
rate

(cid:80)T

E[f ( ¯wT ) − f ∗] ≤

α
T

+

√
√

ασ
T

,

where α =

(cid:18) D2
ηmin

1
2

(cid:19)2

+ 2ηmax

dLmax.

We ﬁrst use the above result to prove Theorems 1 and 2. The proof of Theorem 1 is immediate by
plugging η = ηmin = ηmax in Proposition 1. We recall its statement;

Theorem 1 (Constant step-size AdaGrad). Assuming (i) convexity and (ii) Lmax-
smoothness of each fi, and (iii) bounded iterates, AdaGrad with a constant step-size η
(cid:80)T
and uniform averaging such that ¯wT = 1
T

E[f ( ¯wT ) − f ∗] ≤

α
T

+

√
√

ασ
T

k=1 wk, converges at a rate
(cid:18) D2
η

+ 2η

1
2

(cid:19)2

, where α =

dLmax.

For Theorem 2, we use the properties of the conservative Lipschitz line-search. We recall its statement;

Theorem 2. Under the same assumptions as Theorem 1, AdaGrad with a conservative
Lipschitz line-search with c = 1/2, a step-size upper bound ηmax and uniform averaging
converges at a rate

E[f ( ¯wT ) − f ∗] ≤

α
T

+

√
√

ασ
T

, where α =

(cid:18)

1
2

D2 max

(cid:26) 1

ηmax

(cid:27)

(cid:19)2

, Lmax

+ 2 ηmax

dLmax.

Proof of Theorem 2. Using Lemma 1, there is a step-size ηk that satisﬁes the Lipschitz line-search
with ηk ≥ 2 (1−c)/Lmax. Setting c = 1/2 and using a maximum step-size ηmax, we have

(cid:26)

min

ηmax,

(cid:27)

1
Lmax

≤ ηk ≤ ηmax,

=⇒

1
ηmin

= max

(cid:26) 1

ηmax

(cid:27)

, Lmax

.

Before going into the proof of Proposition 1, we recall some standard lemmas from the adaptive
gradient literature (Theorem 7 & Lemma 10 in (Duchi et al., 2011), Lemma 5.15 & 5.16 in (Hazan,
2016)), and a useful quadratic inequality (Levy et al., 2018, Part of Theorem 4.2)). We include proofs
in Appendix C.1 for completeness.

Lemma 3. If the preconditioners are non-decreasing (Ak (cid:23) Ak−1), the step-sizes are non-
increasing (ηk ≤ ηk−1), and the iterates stay within a ball of radius D of the minima,

(cid:80)T

k=1 (cid:107)wk − w∗(cid:107)2

1
ηk

Ak− 1

ηk−1

Ak−1

≤ D2
ηT

Tr(AT ).

Lemma 4. For AdaGrad, Ak =

(cid:80)T

k=1 (cid:107)∇fik (wk)(cid:107)2

A−1
k

(cid:104)(cid:80)k

i=1 ∇fik (wk)∇fik (wk)(cid:62)(cid:105)1/2
(cid:113)

≤ 2Tr(AT ),

Tr(AT ) ≤

and satisﬁes,

d (cid:80)T

k=1 (cid:107)∇fik (wk)(cid:107)2.

17

Under review as a conference paper at ICLR 2021

Lemma 5. If x2 ≤ a(x + b) for a ≥ 0 and b ≥ 0,

x ≤

(cid:16)(cid:112)

1
2

a2 + 4ab + a

(cid:17)

≤ a +

√

ab.

We now prove Proposition 1.

Proof of Proposition 1. We ﬁrst give an overview of the main steps. Using the deﬁnition of the update
rule, along with Lemmas 3 and 4, we will show that

2 (cid:80)T

k=1(cid:104)∇fik (wk), wk − w∗(cid:105) ≤

(cid:16) D2
ηmin

+ 2ηmax

(cid:17)

Tr(AT ).

(5)

Using the deﬁnition of AT , individual smoothness and convexity, we then show that for a constant a,
(cid:20)(cid:113)(cid:80)T

(cid:80)T

(cid:21)
k=1 fik (wk) − fik (w∗)

(cid:16)
k=1 E[f (wk) − f ∗] ≤ a

E

+ T σ2(cid:17)

,

(6)

Using the quadratic inequality (Lemma 5), averaging and using Jensen’s inequality ﬁnishes the proof.
To derive Eq. (5), we start with the Update rule, measuring distances to w∗ in the (cid:107)·(cid:107)Ak

norm,

(cid:107)wk+1 − w∗(cid:107)2
Ak

= (cid:107)wk − w∗(cid:107)2
Ak

− 2ηk(cid:104)∇fik (wk), wk − w∗(cid:105) + η2

k (cid:107)∇fik (wk)(cid:107)2

A−1
k

.

Dividing by ηk, reorganizing the equation and summing across iterations yields

2

T
(cid:88)

k=1

(cid:104)∇fik (wk), wk − w∗(cid:105) ≤

≤

T
(cid:88)

k=1

T
(cid:88)

k=1

(cid:107)wk − w∗(cid:107)2

(cid:16) Ak
ηk

(cid:17) +

−

Ak−1
ηk−1

T
(cid:88)

k=1

ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

,

(cid:107)wk − w∗(cid:107)2

(cid:16) Ak
ηk

−

Ak−1
ηk−1

(cid:17) + ηmax

T
(cid:88)

k=1

(cid:107)∇fik (wk)(cid:107)2

A−1
k

.

We use the Lemmas 3, 4 to bound the RHS by the trace of the last preconditioner,

≤

≤

D2
ηT
(cid:18) D2
ηmin

Tr(AT ) + 2ηmaxTr(AT ),

(cid:19)

+ 2ηmax

Tr(AT ).

(Lemmas 3 and 4)

(ηk ≥ ηmin)

To derive Eq. (6), we bound the trace of AT using Lemma 4 and Individual Smoothness,

√

√

√

Tr(AT ) ≤

≤

≤

(cid:113)(cid:80)T
d

2dLmax

2dLmax

(cid:113)(cid:80)T

k=1 (cid:107)∇fik (wk)(cid:107)2,
(cid:113)(cid:80)T

k=1 fik (wk) − f ∗
ik

(Lemma 4, Trace bound)

.

(Individual Smoothness)

Combining the above inequalities with δik = fik (w∗) − f ∗

k=1 fik (wk) − fik (w∗) + fik (w∗) − f ∗
ik
2 ( D2

ik and a = 1

ηmin

(±fik (w∗))
√

2dLmax,

+ 2ηmax)

(cid:80)T

k=1(cid:104)∇fik (wk), wk − w∗(cid:105) ≤ a

(cid:113)(cid:80)T

k=1 fik (wk) − fik (w∗) + δik .

Using Individual Convexity and taking expectations,

(cid:80)T

k=1 E[f (wk) − f ∗] ≤ a E
(cid:114)

(cid:20)(cid:113)(cid:80)T

k=1 fik (wk) − fik (w∗) + δik

(cid:21)
,

≤ a

(cid:104)(cid:80)T

k=1 fik (wk) − fik (w∗) + δik

(cid:105)
.

(Jensen’s inequality)

E

Letting σ2 := Ei[δi] = Ei[fi(w∗) − f ∗
(cid:33)2

(cid:32) T

(cid:88)

E[f (wk) − f ∗]

i ] and taking the square on both sides yields
(cid:33)
.

(cid:35)
fik (wk) − fik (w∗)

+ T σ2

≤ a2

(cid:34) T

(cid:88)

(cid:32)

E

k=1

k=1

18

Under review as a conference paper at ICLR 2021

The quadratic bound (Lemma 5) x2 ≤ α(x + β) implies x ≤ α +

√

αβ, with

x =

T
(cid:88)

k=1

E[f (wk) − f ∗],

α =

(cid:18)

1
2

D2 1
ηmin

(cid:19)2

+ 2ηmax

dLmax,

β = T σ2,

gives the ﬁrst bound below. Averaging ¯wT = 1
T

(cid:80)T

k=1wk and using Jensen’s inequality give the result;

E[f (wk) − f ∗] ≤ α + (cid:112)αβ

=⇒

E[f ( ¯wT ) − f ∗] ≤

T
(cid:88)

k=1

α
T

+

√
√

ασ
T

.

19

Under review as a conference paper at ICLR 2021

C.1 PROOFS OF ADAPTIVE GRADIENT LEMMAS

For completeness, we give proofs for the lemmas used in the previous section. We restate them here;

Lemma 3. If the preconditioners are non-decreasing (Ak (cid:23) Ak−1), the step-sizes are
non-increasing (ηk ≤ ηk−1), and the iterates stay within a ball of radius D of the minima,

(cid:80)T

k=1 (cid:107)wk − w∗(cid:107)2

1
ηk

Ak− 1

ηk−1

Ak−1

≤ D2
ηT

Tr(AT ).

Proof of Lemma 3. Under the assumptions that Ak is non-decreasing and ηk is non-increasing,
1
ηk

Ak − 1
ηk−1
(cid:80)T
k=1 (cid:107)wk − w∗(cid:107)2

Ak−1 (cid:23) 0, so we can use the Bounded iterates assumption to bound
(cid:16) Ak
ηk
(cid:16) Ak
ηk

(cid:107)wk − w∗(cid:107)2

≤ D2(cid:80)T

− Ak−1
ηk−1

− Ak−1
ηk−1

k=1 λmax

k=1 λmax

≤ (cid:80)T

Ak−1
ηk−1

Ak
ηk

(cid:17)

(cid:17)

−

.

We then upper-bound λmax by the trace and use the linearity of the trace to telescope the sum,
(cid:16) Ak−1
(cid:16) Ak
ηk−1
ηk

= D2 (cid:80)T

(cid:16) Ak
ηk

k=1 Tr

− Tr

(cid:17)

(cid:17)

(cid:17)
,

≤ D2 (cid:80)T
= D2(cid:16)

Tr

k=1 Tr
(cid:16) AT
(cid:17)
ηT

− Ak−1
ηk−1
(cid:17)(cid:17)
(cid:16) A0
η0

− Tr

≤ D2 1
ηT

Tr(AT ).

Lemma 4. For AdaGrad, Ak =

(cid:104)(cid:80)k

i=1 ∇fik (wk)∇fik (wk)(cid:62)(cid:105)1/2

and satisﬁes,

(cid:80)T

k=1 (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤ 2Tr(AT ),

Tr(AT ) ≤

(cid:113)

d (cid:80)T

k=1 (cid:107)∇fik (wk)(cid:107)2.

A−1
1

Proof of Lemma 4. For ease of notation, let ∇k := ∇fik (wk). By induction, starting with T = 1,
(cid:107)∇fi1(w1)(cid:107)2

1 A−1
= ∇(cid:62)
= Tr(cid:0)A−1

1 ∇1 = Tr(cid:0)∇(cid:62)
(cid:1) = Tr(A1).
1 A2
1
Suppose that it holds for T − 1, (cid:80)T −1
k=1 (cid:107)∇k(cid:107)2
T . Using the deﬁnition of the preconditioner and the cyclic property of the trace,
≤ 2Tr(AT −1) + (cid:107)∇T (cid:107)2

(cid:1), (Cyclic property of trace)
1 )1/2)

≤ 2Tr(AT −1). We will show that it also holds for

(Induction hypothesis)

(A1 = (∇1∇(cid:62)

(cid:1) = Tr(cid:0)A−1

1 ∇1∇(cid:62)
1

k=1 (cid:107)∇fik (wk)(cid:107)2

1 A−1

1 ∇1

A−1
k

(cid:80)T

A−1
k

(cid:16)

= 2Tr

(A2

T − ∇T ∇(cid:62)

A−1
T

T )1/2(cid:17)

+ Tr(cid:0)A−1

T ∇T ∇(cid:62)

T

(cid:1)

(AdaGrad update)

We then use the fact that for any X (cid:23) Y (cid:23) 0, we have (Duchi et al., 2011, Lemma 8)

(cid:16)

(X − Y )1/2(cid:17)

2Tr

+ Tr

(cid:16)

X −1/2Y

(cid:17)

≤ 2Tr

(cid:16)

X 1/2(cid:17)

.

As X = A2

T (cid:23) Y = ∇T ∇(cid:62)

T (cid:23) 0, we can use the above inequality and the induction holds for T .

For the trace bound, recall that AT = G1/2
Jensen’s inequality,

T where GT = (cid:80)T

i=1 ∇fik (wk)∇fik (wk)(cid:62). We use

Tr(AT ) = Tr

(cid:17)

(cid:16)

G

1/2
T

= (cid:80)d

(cid:18)
(cid:112)λj(GT ) = d

(cid:80)d

j=1

(cid:19)
(cid:112)λj(GT )

,

j=1
(cid:113) 1
d

≤ d

(cid:80)d

j=1 λj(GT ) =

d(cid:112)Tr(GT ).

1
d
√

To ﬁnish the proof, we use the deﬁnition of GT and the linearity of the trace to get

(cid:112)Tr(GT ) =

(cid:114)

Tr

(cid:16)(cid:80)T

k=1 ∇k∇k

(cid:62)

(cid:17)

=

(cid:113)(cid:80)T

k=1 Tr(∇k∇k

(cid:62)) =

(cid:113)(cid:80)T

k=1 (cid:107)∇k(cid:107)2.

20

Under review as a conference paper at ICLR 2021

Lemma 5. If x2 ≤ a(x + b) for a ≥ 0 and b ≥ 0,

x ≤

(cid:16)(cid:112)

1
2

a2 + 4ab + a

(cid:17)

≤ a +

√

ab.

Proof of Lemma 5. The starting point is the quadratic inequality x2 − ax − ab ≤ 0. Letting r1 ≤ r2
be the roots of the quadratic, the inequality holds if x ∈ [r1, r2]. The upper bound is then given by
using

a + b ≤

a +

√

√

√

b

√

a +

a2 + 4ab
2

≤

√

a +

a2 +
2

√

4ab

r2 =

= a +

√

ab.

C.2 REGRET BOUND FOR ADAGRAD UNDER INTERPOLATION

In the online convex optimization framework, we consider a sequence of functions fk|T
k=1, chosen
potentially adversarially by the environment. The aim of the learner is to output a series of strategies
wk|T
k=1 before seeing the function fk. After choosing wk, the learner suffers the loss fk(wk) and
observes the corresponding gradient vector ∇fk(wk). They suffer an instantaneous regret rk =
fk(wk) − fk(w) compared to a ﬁxed strategy w. The aim is to bound the cumulative regret,

T
(cid:88)

R(T ) =

[fk(wk) − fk(w∗)]

k=1

√

where w∗ = arg min (cid:80)T
k=1 fk(w) is the best strategy if we had access to the entire sequence
of functions in hindsight. Assuming the functions are convex but non-smooth, AdaGrad obtains
an O(1/
T ) regret bound (Duchi et al., 2011). For online convex optimization, the interpolation
assumption implies that the learner model is powerful enough to ﬁt the entire sequence of functions.
For large over-parameterized models like neural networks, where the number of parameters is of the
order of millions, this is a reasonable assumption for large T .

We ﬁrst recall the update of AdaGrad, at iteration k, the learner decides to play the strategy wk,
suffers loss fk(wk) and uses the gradient feedback ∇fk(wk) to update their strategy as
i=1 ∇fk(wk)∇fk(wk)(cid:62)(cid:105)1/2

k ∇fk(wk), where Ak =

wk+1 = wk − ηA−1

(cid:104)(cid:80)k

.

Now we show that for smooth, convex functions under the interpolation assumption, AdaGrad with a
constant step-size can result in constant regret.

Theorem 6. For a sequence of Lmax-smooth, convex functions fk, assuming the iterates remain
bounded s.t. for all k, (cid:107)wk − w∗(cid:107) ≤ D, AdaGrad with a constant step-size η achieves the following
regret bound,

R(T ) ≤

(cid:18)

1
2

D2 1
η

(cid:19)2

+ 2η

dLmax +

(cid:115)

(cid:18)

1
2

D2 1
η

(cid:19)2

+ 2η

dLmaxσ2

√

T

where σ2 is an upper-bound on fk(w∗) − f ∗
k .

Observe that σ2 is the degree to which interpolation is violated, and if σ2 (cid:54)= 0, R(T ) = O(
T )
matching the regret of (Duchi et al., 2011). However, when interpolation is exactly satisﬁed, σ2 = 0,
and R(T ) = O(1).

√

Proof of Theorem 6. The proof follows that of Proposition 1 which is inspired from (Levy et al.,
2018). For convenience, we repeat the basic steps. Measuring distances to w∗ in the (cid:107)·(cid:107)Ak

norm,

(cid:107)wk+1 − w∗(cid:107)2
Ak

= (cid:107)wk − w∗(cid:107)2
Ak

− 2η(cid:104)∇fk(wk), wk − w∗(cid:105) + η2 (cid:107)∇fk(wk)(cid:107)2

A−1
k

.

21

Under review as a conference paper at ICLR 2021

Dividing by 2η, reorganizing the equation and summing across iterations yields

T
(cid:88)

(cid:104)∇fk(wk), wk − w∗(cid:105) ≤

k=1

T
(cid:88)

k=1

(cid:107)wk − w∗(cid:107)2

(cid:16) Ak

2η −

(cid:17) +

Ak−1
2η

η
2

T
(cid:88)

k=1

(cid:107)∇fk(wk)(cid:107)2

A−1
k

.

By convexity of fk, (cid:104)∇fk(wk), wk − w∗(cid:105) ≥ fk(wk) − fk(w∗). Using the deﬁnition of regret,

R(T ) ≤

T
(cid:88)

k=1

(cid:107)wk − w∗(cid:107)2

(cid:16) Ak

2η −

(cid:17) +

Ak−1
2η

η
2

T
(cid:88)

k=1

(cid:107)∇fk(wk)(cid:107)2

A−1
k

.

We use the Lemmas 3, 4 to bound the RHS by the trace of the last preconditioner,

R(T ) ≤

(cid:18) D2
2η

(cid:19)

+ η

Tr(AT ).

We now bound the trace of AT using Lemma 4 and Individual Smoothness,

(cid:113)(cid:80)T

d

2dLmax

k=1 fk(wk) − f ∗
k ,

k=1 (cid:107)∇fk(wk)(cid:107)2,
(cid:113)(cid:80)T
(cid:113)(cid:80)T

√

√

√

Tr(AT ) ≤

≤

≤

≤

2dLmax

k=1 fk(wk) − fk(w∗) + fk(w∗) − f ∗
k ,

(±fk(w∗))

(cid:112)

2dLmax

(cid:112)R(T ) + σ2T .

(Since fk(w∗) − f ∗

k ≤ σ2)

(Lemma 4, Trace bound)

(Individual Smoothness)

Plugging this back into the regret bound,
(cid:18) D2
2η
(cid:16) D2

Squaring both sides and denoting a =

R(T ) ≤

(cid:19)(cid:112)

2dLmax[(cid:112)R(T ) + σ2T ].

+ η

(cid:17) √

2dLmax,

2η + η

[R(T )]2 ≤ a2[R(T ) + σ2T ].

Using the quadratic bound (Lemma 5) x2 ≤ α(x + β) implies x ≤ α +

x = R(T ),

α =

(cid:18)

1
2

D2 1
η

(cid:19)2

+ 2η

dLmax,

√

αβ, with

β = σ2T,

yields the bound,

R(T ) ≤ α + (cid:112)αβ =

(cid:18)

1
2

D2 1
η

(cid:19)2

+ 2η

dLmax +

(cid:115)

(cid:18)

1
2

D2 1
η

(cid:19)2

+ 2η

dLmaxσ2T .

22

Under review as a conference paper at ICLR 2021

C.3 WITH INTERPOLATION, WITHOUT CONSERVATIVE LINE-SEARCHES

In this section, we show that the conservative constraint ηk+1 ≤ ηk is not necessary if interpolation
is satisﬁed. We give the proof for the Armijo line-search, that has better empirical performance,
but a worse theoretical dependence on the problem’s constants. For the theorem below, amin is
lower-bounded by (cid:15) in practice. A similar proof also works for the Lipschitz line-search.

Theorem 7 (AdaGrad with Armijo line-search under interpolation). Under the same assumptions
of Proposition 1, but without non-increasing step-sizes, if interpolation is satisﬁed, AdaGrad with
the Armijo line-search and uniform averaging converges at the rate,

E[f ( ¯wT ) − f ∗] ≤

(cid:0)D2 + 2η2
max
2T

(cid:1)2

dLmax

(cid:18)

max

(cid:26) 1

ηmax

,

Lmax
amin

(cid:27)(cid:19)2

.

where amin = mink{λmin(Ak)}.

Proof of Theorem 7. Following the proof of Proposition 1,

2

T
(cid:88)

k=1

ηk(cid:104)∇fik (wk), wk − w∗(cid:105) =

T
(cid:88)

k=1

(cid:107)wk − w∗(cid:107)2
Ak

− (cid:107)wk+1 − w∗(cid:107)2
Ak

+ η2

k (cid:107)∇fik (wk)(cid:107)2

A−1
k

.

On the left-hand side, we use individual convexity and interpolation, which implies fik (w∗) =
minw fik (w) and we can bound ηk by ηmin, giving

ηk(cid:104)∇fik (wk), wk − w∗(cid:105) ≥ ηk (fik (wk) − fik (w∗))
(cid:125)

(cid:124)

(cid:123)(cid:122)
≥0

≥ ηmin(fik (wk) − fik (w∗)).

On the right-hand side, we can apply the AdaGrad lemmas (Lemma 4)

+ η2

max (cid:107)∇fik (wk)(cid:107)2

A−1
k

,

T
(cid:88)

(cid:107)wk − w∗(cid:107)2
Ak

− (cid:107)wk+1 − w∗(cid:107)2
Ak

k=1
≤ D2Tr(AT ) + 2η2
(cid:1)√
≤(cid:0)D2 + 2η2

d

max

maxTr(AT ),
(cid:113)(cid:80)T

≤(cid:0)D2 + 2η2

max

(cid:1)√

2dLmax

k=1 (cid:107)∇fik (wk)(cid:107)2,
(cid:113)(cid:80)T

k=1 fik (wk) − fik (w∗).

(By Lemmas 3 and 4)

(By the trace bound of Lemma 4)

Deﬁning a = 1

2ηmin

(cid:0)D2 + 2η2

max

(cid:1)√

2dLmax and combining the previous inequalities yields

(By Individual Smoothness and interpolation)

T
(cid:88)

(fik (wk) − fik (w∗)) ≤ a

k=1 fik (wk) − fik (w∗).

(cid:113)(cid:80)T

Taking expectations and applying Jensen’s inequality yields

k=1

(cid:80)T

k=1 E[f (wk) − f (w∗)] ≤ a

k=1 E[f (wk) − f (w∗)].
k=1 E[f (wk) − f (w∗)], followed by dividing by T and applying

(cid:113)(cid:80)T

Squaring both sides, dividing by (cid:80)T
Jensen’s inequality,

a2
T
Using the Armijo line-search guarantee (Lemma 1) with c = 1/2 and a maximum step-size ηmax,
(cid:26)

E[f ( ¯wT ) − f (w∗)] ≤

max
minT

dLmax

=

(cid:27)

.

(cid:0)D2 + 2η2
2η2

(cid:1)2

ηmin = min

ηmax,

amin
Lmax

,

where amin = mink{λmin(Ak)}, giving the rate
(cid:0)D2 + 2η2
max
2T

E[f ( ¯wT ) − f (w∗)] ≤

(cid:1)2

dLmax

(cid:18)

max

(cid:26) 1

ηmax

,

Lmax
amin

(cid:27)(cid:19)2

.

23

Under review as a conference paper at ICLR 2021

D PROOFS FOR AMSGRAD AND NON-DECREASING PRECONDITIONERS

WITHOUT MOMENTUM

We now give the proofs for AMSGrad and general bounded, non-decreasing preconditioners in the
smooth setting, using a constant step-size (Theorem 8) and the Armijo line-search (Theorem 4). As
in Appendix C, we prove a general proposition and specialize it for each of the theorems;

Proposition 2. In addition to assumptions of Theorem 1, assume that (iv) the preconditioners are
non-decreasing and have (v) bounded eigenvalues in the [amin, amax] range. If the step-sizes are
constrained to lie in the range [ηmin, ηmax] and satisfy

ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤ M (fik (wk) − fik

∗),

for some M < 2,

(7)

using uniform averaging ¯wT = 1
T

(cid:80)T

E[f ( ¯wT ) − f ∗] ≤

k=1 wk leads to the rate
(cid:18) 2

D2damax
(2 − M )ηmin

+

1
T

2 − M

(cid:19)

− 1

σ2.

ηmax
ηmin

Theorem 8. Under the assumptions of Theorem 1 and assuming (iv) non-decreasing precondition-
ers (v) bounded eigenvalues in the [amin, amax] interval, AMSGrad with no momentum, constant
step-size η = amin
2Lmax

and uniform averaging converges at a rate,

E[f ( ¯wT ) − f ∗] ≤

2D2d amax Lmax
amin T

+ σ2.

Proof of Theorem 8. Using Bounded preconditioner and Individual Smoothness, we have that

(cid:107)∇fik (wk)(cid:107)2

1
amin
A constant step-size ηmax = ηmin = amin
2Lmax
(cid:18) 2

A−1
k

≤

1
T

D2damax
(2 − M )ηmin

+

2 − M

2Lmax
amin

(fik (wk) − fik

(cid:107)∇fik (wk)(cid:107)2 ≤
satisﬁes the step-size assumption (Eq. 7) with M = 1 and
2LmaxD2damax
ηmax
amin
ηmin

+ σ2.

σ2 =

− 1

1
T

∗ ).

(cid:19)

We restate Theorem 4;

Theorem 4. Under the same assumptions as Theorem 1, AMSGrad with zero momentum,
Armijo line-search with c = 3/4, a step-size upper bound ηmax and uniform averaging
converges at a rate,

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 3D2d · amax
2T

+ 3ηmaxσ2

(cid:19)

max

(cid:26) 1

ηmax

,

2Lmax
amin

(cid:27)

.

Proof of Theorem 4. For the Armijo line-search, Lemma 1 guarantees that
η (cid:107)∇fik (wk)(cid:107)2

(fik (wk) − f ∗
ik

and min

ηmax,

≤

(cid:26)

),

A−1
k

2 λmin(Ak) (1 − c)
Lmax

1
c

(cid:27)

≤ η ≤ ηmax.

Selecting c = 3/4 gives M = 4/3 and ηmin = min

(cid:110)

ηmax, amin
2Lmax

(cid:111)

, so

1
T

D2damax
(2 − M )ηmin

+

(cid:18) 2

2 − M

ηmax
ηmin

(cid:19)

σ2

− 1

(cid:18) 2

+

2 − 4/3

ηmax
ηmin
(cid:19)

− 1

σ2,

=

=

≤

D2damax
1
(2 − 4/3)ηmin
T
3D2damax
1
T
2ηmin
3D2damax
2T

max

+

(cid:18) 3ηmax
ηmin
(cid:26) 1

ηmax

(cid:19)

σ2,

− 1

(cid:27)

,

2Lmax
amin

+ 3ηmaxσ2 max

(cid:26) 1

ηmax

,

2Lmax
amin

(cid:27)

.

24

Under review as a conference paper at ICLR 2021

Theorem 9. Under the assumptions of Theorem 1 and assuming (iv) non-decreasing precondition-
ers (v) bounded eigenvalues in the [amin, amax] interval, AMSGrad with no momentum, Armijo
SPS with c = 3/4 and uniform averaging converges at a rate,

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 3D2d · amax
2T

+ 3ηmaxσ2

(cid:19)

max

(cid:26) 1

ηmax

,

3Lmax
2amin

(cid:27)

.

Proof of Theorem 5. For Armijo SPS, Lemma 2 guarantees that
(cid:26)

ηk (cid:107)∇fik (wk)(cid:107)2

≤

A−1
k

(fik (wk) − f ∗
ik

),

and

min

ηmax,

1
c

Selecting c = 3/4 gives M = 4/3 and ηmin = min

(cid:110)

ηmax, 2amin
3Lmax

(cid:111)

, so

1
T

D2damax
(2 − M )ηmin

+

(cid:18) 2

2 − M

ηmax
ηmin

(cid:19)

σ2

− 1

(cid:27)

amin
2c Lmax

≤ η ≤ ηmax.

(cid:19)

σ2,

− 1

(cid:18) 2

+

2 − 4/3

ηmax
ηmin
(cid:19)

− 1

σ2,

=

=

≤

D2damax
1
(2 − 4/3)ηmin
T
3D2damax
1
T
2ηmin
3D2damax
2T

max

+

(cid:18) 3ηmax
ηmin
(cid:26) 1

ηmax

(cid:27)

,

3Lmax
2amin

+ 3ηmaxσ2 max

(cid:26) 1

ηmax

,

3Lmax
2amin

(cid:27)

.

Before diving into the proof of Proposition 2, we prove the following lemma to handle terms of the
form ηk(fik (wk) − fik (w∗)). If ηk depends on the function sampled at the current iteration, fik , as
in the case of line-search, we cannot take expectations as the terms are not independent. Lemma 6
bounds ηk(fik (wk) − fik (w∗)) in terms of the range [ηmin, ηmax];
Lemma 6. If 0 ≤ ηmin ≤ η ≤ ηmax and the minimum value of fi is f ∗

i , then

η(fi(w) − fi(w∗)) ≥ ηmin(fi(w) − fi(w∗)) − (ηmax − ηmin)(fi(w∗) − f ∗

i ).

Proof of Lemma 6. By adding and subtracting f ∗
i , the minimum value of fi, we get a non-negative
and a non-positive term multiplied by η. We can use the bounds η ≥ ηmin and η ≤ ηmax separately;

η[fi(w) − fi(w∗)] = η[fi(w) − f ∗
i
(cid:125)

(cid:124)

(cid:123)(cid:122)
≥0
≥ ηmin[fi(w) − f ∗

+ f ∗
(cid:124)

i − fi(w∗)
],
(cid:125)
(cid:123)(cid:122)
≤0
i ] + ηmax[f ∗

i − fi(w∗)].

Adding and subtracting ηminfi(w∗) ﬁnishes the proof,

= ηmin[fi(w) − fi(w∗) + fi(w∗) − f ∗
= ηmin[fi(w) − fi(w∗)] + (ηmax − ηmin)[f ∗

i ] + ηmax[f ∗

i − fi(w∗)].

i − fi(w∗)],

Proof of Proposition 2. We start with the Update rule, measuring distances to w∗ in the (cid:107)·(cid:107)Ak

norm,

(cid:107)wk+1 − w∗(cid:107)2
Ak

= (cid:107)wk − w∗(cid:107)2
Ak

− 2ηk(cid:104)∇fik (wk), wk − w∗(cid:105) + η2

k (cid:107)∇fik (wk)(cid:107)2

A−1
k

(8)

To bound the RHS, we use the assumption on the step-sizes (Eq. (7)) and Individual Convexity,
k (cid:107)∇fik (wk)(cid:107)2

− 2ηk(cid:104)∇fik (wk), wk − w∗(cid:105) + η2
∗),
≤ −2ηk(cid:104)∇fik (wk), wk − w∗(cid:105) + M ηk(fik (wk) − fik
∗ ),
≤ −2ηk[fik (wk) − fik (w∗)] + M ηk(fik (wk) − fik
∗),
≤ −2ηk[fik (wk) − fik (w∗)] + M ηk(fik (wk) − fik (w∗) + fik (w∗) − fik
∗).
≤ −(2 − M )ηk[fik (wk) − fik (w∗)] + M ηmax(fik (w∗) − fik

(Step-size assumption, Eq. (7))
(Individual Convexity)
(±fik (w∗))
(ηk ≤ ηmax)

A−1
k

,

25

Under review as a conference paper at ICLR 2021

Plugging the inequality back into Eq. (8) and reorganizing the terms yields

(2 − M )ηk[fik (wk) − fik (w∗)] ≤

(cid:16)

− (cid:107)wk+1 − w∗(cid:107)2
(cid:107)wk − w∗(cid:107)2
Ak
Ak
∗ )
+ M ηmax(fik (w∗) − fik

(cid:17)

(9)

Using Lemma 6, we have that

(2 − M )ηk[fik (wk) − fik (w∗)] ≥ (2 − M )ηmin(fik (wk) − fik (w∗))

∗).
− (2 − M )(ηmax − ηmin)(fik (w∗) − fik

Using this inequality in Eq. (9), we have that

∗)
(2 − M )ηmin(fik (wk) − fik (w∗)) − (2 − M )(ηmax − ηmin)(fik (w∗) − fik
∗),
+ M ηmax(fik (w∗) − fik

− (cid:107)wk+1 − w∗(cid:107)2
Ak

(cid:107)wk − w∗(cid:107)2
Ak

≤

(cid:16)

(cid:17)

Moving the terms depending on fik (w∗) − fik

(2 − M )ηmin(fik (wk) − fik (w∗)) ≤

∗ to the RHS,
(cid:16)

(cid:107)wk − w∗(cid:107)2
Ak
∗).
+ (2ηmax − (2 − M )ηmin)(fik (w∗) − fik

− (cid:107)wk+1 − w∗(cid:107)2
Ak

(cid:17)

Taking expectations and summing across iterations yields

(2 − M )ηmin

T
(cid:88)

k=1

E[fik (wk) − fik (w∗)] ≤ E

(cid:34) T

(cid:88)

(cid:16)

(cid:107)wk − w∗(cid:107)2
Ak

− (cid:107)wk+1 − w∗(cid:107)2
Ak

(cid:35)

(cid:17)

k=1

+(2ηmax − (2 − M )ηmin)T σ2.

Using Lemma 3 to telescope the distances and using the Bounded preconditioner,

T
(cid:88)

k=1

(cid:107)wk − w∗(cid:107)2
Ak

− (cid:107)wk+1 − w∗(cid:107)2
Ak

≤

T
(cid:88)

k=1

(cid:107)wk − w∗(cid:107)2

Ak−Ak−1

≤ D2 Tr(AT ) ≤ D2 d amax,

which guarantees that

(2 − M )ηmin

T
(cid:88)

k=1

E[f (wk) − f (w∗)] ≤D2damax + (2ηmax − (2 − M )ηmin)T σ2.

Dividing by T (2 − M )ηmin and using Jensen’s inequality ﬁnishes the proof, giving the rate for the
averaged iterate,

E[f ( ¯wT ) − f (w∗)] ≤

1
T

D2damax
(2 − M )ηmin

+

(cid:18) 2

2 − M

ηmax
ηmin

(cid:19)

− 1

σ2.

26

Under review as a conference paper at ICLR 2021

E AMSGRAD WITH MOMENTUM

We ﬁrst show the relation between the AMSGrad momentum and heavy ball momentum and then
present the proofs with AMSGrad momentum in E.2 and heavy ball momentum in E.3.

E.1 RELATION BETWEEN THE AMSGRAD UPDATE AND PRECONDITIONED SGD WITH

HEAVY-BALL MOMENTUM

Recall that the AMSGrad update is given as:

wk+1 = wk − ηk A−1

k mk

; mk = βmk−1 + (1 − β)∇fik (wk)

Simplifying,

wk+1 = wk − ηk A−1
wk+1 = wk − ηk(1 − β) A−1

k (βmk−1 + (1 − β)∇fik (wk))
k ∇fik (wk) − ηkβ A−1

k mk−1

From the update at iteration k − 1,

wk = wk−1 − ηk−1 A−1

k−1mk−1

=⇒ −mk−1 =

1
ηk−1

Ak−1 (wk − wk−1)

From the above relations,

wk+1 = wk − ηk(1 − β) A−1

k ∇fik (wk) + β

ηk
ηk−1

A−1

k Ak−1 (wk − wk−1)

which is of the same form as

wk+1 = wk − ηk A−1

k + γ(wk − wk−1),

the update with heavy ball momentum. The two updates are equivalent up to constants except for the
key difference that for AMSGrad, the momentum vector (wk − wk−1) is further preconditioned by
A−1

k Ak−1.

27

Under review as a conference paper at ICLR 2021

E.2 PROOFS FOR AMSGRAD WITH MOMENTUM

We now give the proofs for AMSGrad having the update.

wk+1 = wk − ηk A−1

k mk

; mk = βmk−1 + (1 − β)∇fik (wk)

We analyze it in the smooth setting using a constant step-size (Theorem 3), conservative Armijo
SPS (Theorem 5) and conservative Armijo SLS (Theorem 10). As before, we abstract the common
elements to a general proposition and specialize it for each of the theorems.

Proposition 3. In addition to assumptions of Theorem 1, assume that (iv) the preconditioners are
non-decreasing and have (v) bounded eigenvalues in the [amin, amax] range. If the step-sizes are
lower-bounded and non-increasing, ηmin ≤ ηk ≤ ηk−1 and satisfy

ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤ M (fik (wk) − fik

∗),

for some M < 2

1 − β
1 + β

,

(10)

using uniform averaging ¯wT = 1
T

(cid:80)T

k=1 wk leads to the rate

E[f ( ¯wT ) − f ∗] ≤

(cid:18)

2 −

1 + β
1 − β

1 + β
1 − β

M

(cid:19)−1(cid:20) D2damax

ηminT

+ M σ2

(cid:21)
.

We ﬁrst show how the convergence rate of each step-size method can be derived from Proposition 3.

Theorem 3. Under the same assumptions as Theorem 1, and assuming (iv) non-
decreasing preconditioners (v) bounded eigenvalues in the [amin, amax] interval, where
κ = amax/amin, AMSGrad with β ∈ [0, 1), constant step-size η = 1−β
and uniform
1+β
averaging converges at a rate,

amin
2Lmax

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 1 + β
1 − β

(cid:19)2 2LmaxD2dκ
T

+ σ2.

Proof of Theorem 3. Using Bounded preconditioner and Individual Smoothness, we have that

η (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤ η

1
amin

(cid:107)∇fik (wk)(cid:107)2 ≤ η

2Lmax
amin

(fik (wk) − fik

∗).

Using a constant step-size η = 1−β
1+β
constant M = 1−β

1+β . The convergence is then,

amin
2Lmax

satisﬁes the requirement of Proposition 3 (Eq. (10)) with

E[f ( ¯wT ) − f (w∗)] ≤

=

1 + β
1 − β

1 + β
1 − β

(cid:34)

(cid:18)

2 −

1 + β
1 − β

M

(cid:19)−1(cid:20) D2damax

(cid:21)

+ M σ2,

ηminT
(cid:35)

σ2,

1 − β
1 + β

+ σ2,

+

D2damax
1−β
amin
T
2Lmax
1+β
(cid:19)2 2LmaxD2dκ
T

(cid:18) 1 + β
1 − β

=

with κ = amax/amin.

28

Under review as a conference paper at ICLR 2021

Theorem 5. Under the same assumptions of Theorem 1 and assuming (iv) non-
decreasing preconditioners (v) bounded eigenvalues in the [amin, amax] interval with
κ = amax/amin, AMSGrad with β ∈ [0, 1), conservative Armijo SPS with c = 1+β/1−β
and uniform averaging converges at a rate,

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 1 + β
1 − β

(cid:19)2 2LmaxD2dκ
T

+ σ2.

Proof of Theorem 5. For Armijo SPS, Lemma 2 guarantees that

ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤

1
c

(fik (wk) − f ∗
ik

),

and

amin
2c Lmax

≤ ηk.

Setting c = 1+β
1−β
amin
2Lmax
1+β

1−β ensures that M = 1/c satisﬁes the requirement of Proposition 3 and ηmin ≥

. Plugging in these values into Proposition 3 completes the proof.

Theorem 10. Under the assumptions of Theorem 1 and assuming (iv) non-decreasing precon-
ditioners (v) bounded eigenvalues in the [amin, amax] interval, AMSGrad with momentum with
1+β
parameter β ∈ [0, 1/5), conservative Armijo SLS with c = 2
1−β and uniform averaging converges
3
at a rate,

E[f ( ¯wT ) − f ∗] ≤ 3

1 + β
1 − 5β

LmaxD2dκ
T

+ 3σ2

Proof of Theorem 10. For Armijo SLS, Lemma 1 guarantees that

ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤

1
c

(fik (wk) − f ∗
ik

),

and

2(1 − c) amin
Lmax

≤ ηk.

The line-search parameter c is restricted to [0, 1] and relates to the the requirement parameter M
of Proposition 3 (Eq. (10)) through M = 1/c. The combined requirements on M are then that
1 < M < 2 1−β
3 . To leave room to satisfy the constraints, let β < 1
5 .

1+β , which is only feasible if β < 1

Setting 1

c = M = 3

2

1−β
1+β satisﬁes the constraints and requirement for Proposition 3, and

E[f ( ¯wT ) − f (w∗)] ≤

=

=

1 + β
1 − β

1 + β
1 − β
1 + β
1 − β

(cid:18)

2 −

1 + β
1 − β

M

(cid:19)−1(cid:20) D2damax

ηminT

+ M σ2

(cid:21)
,

(cid:18)

2 −

3
2
Lmax
(1 − c)

(cid:19)−1(cid:20)

Lmax
2(1 − c) amin

D2dκ
T

+ 3σ2 = 3

+

3
2

1 − β
1 + β

(cid:21)
,

σ2

D2damax
T
1 + β
1 − 5β

LmaxD2dκ
T

+ 3σ2.

where the last step substituted 1/(1 − c),

1 − c = 1 −

2
3

1 + β
1 − β

=

3(1 − β) − 2(1 + β)
3(1 − β)

=

1
3

1 − 5β
1 − β

.

Before diving into the proof of Proposition 3, we prove the following lemma,

Lemma 7. For any set of vectors a, b, c, d, if a = b + c, then,

(cid:107)a − d(cid:107)2 = (cid:107)b − d(cid:107)2 − (cid:107)a − b(cid:107)2 + 2(cid:104)c, a − d(cid:105)

29

Under review as a conference paper at ICLR 2021

Proof.

(cid:107)a − d(cid:107)2 = (cid:107)b + c − d(cid:107)2 = (cid:107)b − d(cid:107)2 + 2(cid:104)c, b − d(cid:105) + (cid:107)c(cid:107)2

Since c = a − b,

= (cid:107)b − d(cid:107)2 + 2(cid:104)a − b, b − d(cid:105) + (cid:107)a − b(cid:107)2
= (cid:107)b − d(cid:107)2 + 2(cid:104)a − b, b − a + a − d(cid:105) + (cid:107)a − b(cid:107)2
= (cid:107)b − d(cid:107)2 + 2(cid:104)a − b, b − a(cid:105) + 2(cid:104)a − b, a − d(cid:105) + (cid:107)a − b(cid:107)2
= (cid:107)b − d(cid:107)2 − 2 (cid:107)a − b(cid:107)2 + 2(cid:104)a − b, a − d(cid:105) + (cid:107)a − b(cid:107)2
= (cid:107)b − d(cid:107)2 − (cid:107)a − b(cid:107)2 + 2(cid:104)c, a − d(cid:105)

We now move to the proof of the main proposition. Our proof follows the structure of Reddi et al.
(2018); Alacaoglu et al. (2020).

Proof of Proposition 3. To reduce clutter, let Pk = Ak/ηk. Using the update, we have the expansion
(cid:1) − w∗,
k ∇fik (wk) − βP −1

wk+1 − w∗ = (cid:0)wk − P −1

= (cid:0)wk − (1 − β)P −1

(cid:1) − w∗,

k mk−1

k mk

Measuring distances in the (cid:107)·(cid:107)Pk

-norm, such that (cid:107)x(cid:107)2
Pk

= (cid:104)x, Pkx(cid:105),

(cid:107)wk+1 − w∗(cid:107)2
Pk

= (cid:107)wk − w∗(cid:107)2
Pk

− 2(1 − β) (cid:104)wk − w∗, ∇fik (wk)(cid:105),
− 2β (cid:104)wk − w∗, mk−1(cid:105) + (cid:107)mk(cid:107)2

P −1
k

.

We separate the distance to w∗ from the momentum in the second inner product using the update and
Lemma 7 with a = c = P 1/2

k−1(wk − w∗), b = 0, d = P 1/2

k−1(wk−1 − w∗).

−2(cid:104)mk−1, wk − w∗(cid:105) = −2 (cid:104)Pk−1(wk−1 − wk), wk − w∗(cid:105),
+ (cid:107)wk − w∗(cid:107)2

(cid:104)
(cid:107)wk − wk−1(cid:107)2

=

= (cid:107)mk−1(cid:107)2
≤ (cid:107)mk−1(cid:107)2

P −1
k−1

P −1
k−1

Pk−1
+ (cid:107)wk − w∗(cid:107)2
+ (cid:107)wk − w∗(cid:107)2
Pk

Pk−1

− (cid:107)wk−1 − w∗(cid:107)2

Pk−1

Pk−1
− (cid:107)wk−1 − w∗(cid:107)2

,

Pk−1

(cid:105)

,

− (cid:107)wk−1 − w∗(cid:107)2

Pk−1

,

where the last inequality uses the fact that ηk ≤ ηk−1 and Ak (cid:23) Ak−1, which implies Pk (cid:23) Pk−1,
and (cid:107)wk − w∗(cid:107)2

. Plugging this inequality in and grouping terms yields

≤ (cid:107)wk − w∗(cid:107)2
Pk

Pk−1

2(1 − β) (cid:104)wk − w∗, ∇fik (wk)(cid:105) ≤

(cid:104)

(cid:107)wk − w∗(cid:107)2
Pk

− (cid:107)wk+1 − w∗(cid:107)2
Pk

(cid:105)

+ β
(cid:104)
+

(cid:104)

(cid:107)wk − w∗(cid:107)2
Pk

− (cid:107)wk−1 − w∗(cid:107)2

Pk−1

(cid:105)

β (cid:107)mk−1(cid:107)2

P −1
k−1

+ (cid:107)mk(cid:107)2

P −1
k

(cid:105)

By convexity, the inner product on the left-hand-side is bounded by (cid:104)wk − w∗, ∇fik (wk)(cid:105) ≥
fik (wk) − fik (w∗). The ﬁrst two lines of the right-hand-side will telescope if we sum all iterations,
so we only need to treat the norms of the momentum terms. We introduce a free parameter δ ≥ 0,
that is only used for the analysis, and expand

β (cid:107)mk−1(cid:107)2

P −1
k−1

+ (cid:107)mk(cid:107)2

P −1
k

= β (cid:107)mk−1(cid:107)2

P −1
k−1

+ (1 + δ) (cid:107)mk(cid:107)2

P −1
k

− δ (cid:107)mk(cid:107)2

P −1
k

.

To bound (cid:107)mk(cid:107)2

P −1
k

, we expand it by its update and use Young’s inequality to get

(cid:107)mk(cid:107)2

P −1
k

= (cid:107)βmk−1 + (1 − β)∇fik (wk)(cid:107)2
≤ (1 + (cid:15))β2 (cid:107)mk−1(cid:107)2

P −1
k

P −1
k

+ (1 + 1/(cid:15))(1 − β)2 (cid:107)∇fik (wk)(cid:107)2

,

P −1
k

30

Under review as a conference paper at ICLR 2021

where (cid:15) > 0 is also a free parameter, introduced to control the tradeoff of the bound. Plugging this
bound in the momentum terms, we get
β (cid:107)mk−1(cid:107)2

+ (1 + (cid:15))(1 + δ)β2 (cid:107)mk−1(cid:107)2

≤ β (cid:107)mk−1(cid:107)2

− δ (cid:107)mk(cid:107)2

+ (cid:107)mk(cid:107)2

,

P −1
k−1

P −1
k

P −1
k−1

P −1
k

P −1
k

+ (1 + 1/(cid:15))(1 + δ)(1 − β)2 (cid:107)∇fik (wk)(cid:107)2

P −1
k

.

As P −1

k (cid:22) P −1

k−1, we have that (cid:107)mk−1(cid:107)2

P −1
k

≤ (cid:107)mk−1(cid:107)2

P −1
k−1

which implies

≤ (cid:0)β + (1 + (cid:15))(1 + δ)β2(cid:1) (cid:107)mk−1(cid:107)2

P −1
k−1

− δ (cid:107)mk(cid:107)2

P −1
k

+ (1 + 1/(cid:15))(1 + δ)(1 − β)2 (cid:107)∇fik (wk)(cid:107)2

P −1
k

.

To get a telescoping sum, we set δ to be equal to β + (1 + (cid:15))(1 + δ)β2, which is satisﬁed if
δ = β+(1+(cid:15))β2

1−(1+(cid:15))β2 , and δ > 0 is satisﬁed if β < 1/√

1+(cid:15). We now plug back the inequality

β (cid:107)mk−1(cid:107)2

P −1
k−1

+ (cid:107)mk(cid:107)2

P −1
k

in the previous expression to get

(cid:104)

≤ δ

(cid:107)mk−1(cid:107)2

− (cid:107)mk(cid:107)2
+ (1 + 1/(cid:15))(1 + δ)(1 − β)2 (cid:107)∇fik (wk)(cid:107)2

P −1
k−1

P −1
k

(cid:105)

,

P −1
k

2(1 − β) (fik (wk) − fik (w∗)) ≤ (cid:107)wk − w∗(cid:107)2
Pk
(cid:107)wk − w∗(cid:107)2
Pk
− (cid:107)mk(cid:107)2

+ β
(cid:104)
(cid:107)mk−1(cid:107)2

+ δ

(cid:104)

− (cid:107)wk+1 − w∗(cid:107)2
Pk

(cid:105)

− (cid:107)wk−1 − w∗(cid:107)2

P −1
k−1

P −1
k

(cid:105)

Pk−1

All terms now telescope, except the gradient norm which we bound using the step size assumption,

+ (1 + 1/(cid:15))(1 + δ)(1 − β)2 (cid:107)∇fik (wk)(cid:107)2

P −1
k

.

(cid:107)∇fik (wk)(cid:107)2

P −1
k

This gives the expression

= ηk (cid:107)∇fik (wk)(cid:107)2
∗).
= M (fik (wk) − fik (w∗)) + M (fik (w∗) − fik

≤ M (fik (wk) − fik

A−1
k

∗),

− (cid:107)wk+1 − w∗(cid:107)2
Pk

α (fik (wk) − fik (w∗)) ≤ (cid:107)wk − w∗(cid:107)2
Pk
(cid:107)wk − w∗(cid:107)2
Pk
− (cid:107)mk(cid:107)2
∗),
+ (1 + 1/(cid:15))(1 + δ)(1 − β)2M (fik (w∗) − fik

− (cid:107)wk−1 − w∗(cid:107)2

(cid:107)mk−1(cid:107)2

+ β
(cid:104)

P −1
k−1

P −1
k

+ δ

Pk−1

(cid:105)

(cid:104)

(cid:105)

with α = 2(1 − β) − (1 + 1/(cid:15))(1 + δ)(1 − β)2M . Summing all iterations, the individual terms are
bounded by the Bounded iterates and Lemma 3;

T
(cid:88)

k=1

T
(cid:88)

k=1

T
(cid:88)

k=1

β

δ

(cid:107)wk − w∗(cid:107)2
Pk

− (cid:107)wk+1 − w∗(cid:107)2
Pk

≤ D2Tr(PT )

(cid:107)wk − w∗(cid:107)2
Pk

− (cid:107)wk−1 − w∗(cid:107)2

Pk−1

≤ β (cid:107)wT − w∗(cid:107)2
PT

≤

D2
ηmin

Tr(AT )

≤ β

D2
ηmin

Tr(AT )

(cid:107)mk−1(cid:107)2

P −1
k−1

− (cid:107)mk(cid:107)2

P −1
k

≤ δ (cid:107)m0(cid:107)2
P0

= 0.

Using the boundedness of the preconditioners gives Tr(AT ) ≤ damax and the total bound

T
(cid:88)

α

(fik (wk) − fik (w∗)) ≤

k=1

(1 + β)D2damax
ηmin

+ (1 + 1/(cid:15))(1 + δ)(1 − β)2M

T
(cid:88)

k=1

∗ ).
(fik (w∗) − fik

31

β + (1 + (cid:15))β2
1 − (1 + (cid:15))β2 > 0,
√

1 + (cid:15). To simplify the

Under review as a conference paper at ICLR 2021

Taking expectations,

α

T
(cid:88)

k=1

E[f (wk) − f (w∗)] ≤

(1 + β)D2damax
ηmin

+ (1 + 1/(cid:15))(1 + δ)(1 − β)2M σ2T.

It remains to expand α and simplify the constants. We had deﬁned

α = 2(1 − β) − (1 + 1/(cid:15))(1 + δ)(1 − β)2M > 0,

and

δ =

where (cid:15) > 0 is a free parameter. This puts the requirement on β that β < 1/
bounds, we set β = 1/(1 + (cid:15)), (cid:15) = 1/β − 1, which gives the substitutions

1 + (cid:15) =

1
β

1 +

1
(cid:15)

=

1
1 − β

δ = 2

β
1 − β

1 + δ =

1 + β
1 − β

.

Plugging those into the rate gives

α

T
(cid:88)

k=1

E[f (wk) − f (w∗)] ≤

(1 + β)D2damax
ηmin

+ (1 + β)M σ2T,

while plugging them into α gives

α = 2(1 − β) − (1 + 1/(cid:15))(1 + δ)(1 − β)2M,
1 + β
1 − β

= (1 − β)

2 −

M

(cid:20)

(cid:21)

, which is positive if M < 2

1 − β
1 + β

.

Dividing by αT , using Jensen’s inequality and averaging ﬁnishes the proof, with the rate

T
(cid:88)

k=1

E[f (wk) − f (w∗)] ≤

(cid:18)

2 −

1 + β
1 − β

1 + β
1 − β

M

(cid:19)−1(cid:20) D2damax

ηminT

+ M σ2

(cid:21)
.

32

Under review as a conference paper at ICLR 2021

E.3 PROOFS FOR AMSGRAD WITH HEAVY BALL MOMENTUM

We now give the proofs for AMSGrad with heavy ball momentum with the update.

wk+1 = wk − ηk A−1

k ∇fik (wk) + γ (wk − wk−1)

We analyze it in the smooth setting using a constant step-size (Theorem 11), a conservative Armijo
SPS (Theorem 12) and conservative Armijo SLS (Theorem 13). As before, we abstract the common
elements to a general proposition and specialize it for each of the theorems.

Proposition 4. In addition to assumptions of Theorem 1, assume that (iv) the preconditioners are
non-decreasing and have (v) bounded eigenvalues in the [amin, amax] range. If the step-sizes are
lower-bounded and non-increasing, ηmin ≤ ηk ≤ ηk−1 and satisfy

ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤ M (fik (wk) − fik

∗ ),

for some M < 2 − 2γ,

(11)

AMSGrad with heavy ball momentum with parameter γ < 1 and uniform averaging ¯wT =
1
T

k=1 wk leads to the rate

(cid:80)T

E[f ( ¯wT ) − f ∗] ≤

1
2 − 2γ − M

(cid:20) 1
T

(cid:18) 2(1 + γ2)D2amaxd
ηmin

+ 2γ[f (w0) − f (w∗)]

(cid:19)

+ M σ2

(cid:21)
.

We ﬁrst show how the convergence rate of each step-size method can be derived from Proposition 4.

Theorem 11. Under the assumptions of Theorem 1 and assuming (iv) non-decreasing precondi-
tioners (v) bounded eigenvalues in the [amin, amax] range, AMSGrad with heavy ball momentum
with parameter γ ∈ [0, 1), constant step-size η = 2amin (1−γ)
and uniform averaging converges at
a rate

3Lmax

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 9
2

1
T

1 + γ2
(1 − γ)2 Lmax D2κd +

3γ
(1 − γ)

[f (w0) − f (w∗)]

(cid:19)

+ 2σ2.

Proof of Theorem 11. Using Bounded preconditioner and Individual Smoothness, we have that

η (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤ η

1
amin

(cid:107)∇fik (wk)(cid:107)2 ≤ η

2Lmax
amin

(fik (wk) − fik

∗).

A constant step-size η = 2amin (1−γ)/3Lmax means the requirement for Proposition 4 is satisﬁed with
M = 4

3 (1 − γ) in Proposition 4 ﬁnishes the proof.

3 (1 − γ). Plugging (2 − 2γ − M ) = 2

Theorem 12. Under the assumptions of Theorem 1 and assuming (iv) non-decreasing precondi-
tioners (v) bounded eigenvalues in the [amin, amax] interval, AMSGrad with heavy ball momentum
with parameter γ ∈ [0, 1), conservative Armijo SPS with c = 3/4(1−γ) and uniform averaging
converges at a rate,

E[f ( ¯wT ) − f ∗] ≤

(cid:18) 9
2

1
T

1 + γ2
(1 − γ)2 LmaxD2κd +

3γ
(1 − γ)

[f (w0) − f (w∗)]

(cid:19)

+ 2σ2.

Proof of Theorem 12. For Armijo SPS, Lemma 2 guarantees that
1
c

ηk (cid:107)∇fik (wk)(cid:107)2

(fik (wk) − f ∗
ik

A−1
k

and

≤

),

amin
2c Lmax

≤ ηk.

Selecting c = 3/4(1−γ) gives M = 4/3(1 − γ) ≤ 2(1 − γ) and the requirement of Proposition 4 are
satisﬁed. The minimum step-size is then ηmin = amin
, so ηmin and M are the same
2cLmax
as in the constant step-size case (Theorem 11) and the same rate applies.

= 2amin (1−γ)
3Lmax

Theorem 13. Under the assumptions of Theorem 1 and assuming (iv) non-decreasing precondi-
tioners (v) bounded eigenvalues in the [amin, amax] interval, AMSGrad with heavy ball momentum
with parameter γ ∈ [0, 1/4), conservative Armijo SLS with c = 3/4(1−γ) and uniform averaging

33

Under review as a conference paper at ICLR 2021

converges at a rate,

E[f ( ¯wT ) − f ∗] ≤

(cid:18)

6

1 + γ2
1 − 4γ

1
T

LmaxD2κd +

3γ
(1 − γ)

[f (w0) − f (w∗)]

(cid:19)

+ 2σ2.

Proof of Theorem 13. Selecting c = 3/4(1−γ) is feasible if γ < 1/4 as c < 1. The Armijo SLS
(Lemma 1) then guarantees that

ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤

1
c

(fik (wk) − f ∗
ik

),

which satisﬁes the requirements of Proposition 4 with M = 4

(cid:18)
6

1 + γ2
1 − γ

D2amaxd
ηmin

+

With c = 3/4

E[f ( ¯wT ) − f (w∗)] ≤

1
T
1−γ , ηmin ≥ 2(1−c)amin
Lmax
(cid:18)

E[f ( ¯wT ) − f (w∗)] ≤

= 2amin
Lmax

1−4γ
4(1−γ) . Plugging it into the above bound yields

1
T

1 + γ2
1 − 4γ

LmaxD2κd +

3γ
(1 − γ)

[f (w0) − f (w∗)]

(cid:19)

+ 2σ2.

6

and

≤ η,

2(1 − c) amin
Lmax
3 (1 − γ). Plugging M in the rate yields
3γ
(1 − γ)

[f (w0) − f (w∗)]

+ 2σ2,

(cid:19)

We now move to the proof of the main proposition. Our proof follows the structure of Ghadimi et al.
(2015); Sebbouh et al. (2020).

Proof of Proposition 4. Recall the update for AMSGrad with heavy-ball momentum,

wk+1 = wk − ηkA−1

k ∇fik (wk) + γ(wk − wk−1).

The proof idea is to analyze the distance from w∗ to wk and a momentum term,

(cid:107)δk(cid:107)2 = (cid:107)wk + mk − w∗(cid:107)2
Ak

,

where mk = γ

1−γ (wk − wk−1),

(12)

(13)

by considering the momentum update (Eq. 12) as a preconditioned step on the joint iterates (wk +mk),

wk+1 + mk+1 = wk + mk − ηk

1−γ A−1

k ∇fik (wk).

(14)

Let us verify Eq. (14). First, expressing wk+1 + mk+1 as a weighted difference of wk+1 and wk,
1−γ (wk+1 − wk) = 1

wk+1 + mk+1 = wk+1 + γ

1−γ wk+1 − γ

1−γ wk.

Expanding wk+1 in terms of the update rule then gives

= 1

= 1

1−γ (wk − ηkA−1
1−γ (wk − ηkA−1
1−γ wk − γ

k ∇fik (wk) + γ(wk − wk−1)) − γ
k ∇fik (wk) − γwk−1),
1−γ A−1

k ∇fik (wk),

= 1

1−γ wk,

1−γ wk−1 − ηk
1−γ A−1

which can then be re-written as wk + mk − ηk
follows similar steps as the analysis without momentum. Using Eq. (14), we have the recurrence

k ∇fik (wk). The analysis of the method then

(cid:107)δk+1(cid:107)2
Ak

= (cid:107)wk+1 + mk+1 − w∗(cid:107)2
Ak

=

(cid:13)
(cid:13)wk + mk − ηk
(cid:13)

1−γ A−1

k ∇fik (wk) − w∗(cid:13)
2
(cid:13)
(cid:13)

Ak

,

= (cid:107)δk(cid:107)2
Ak

(1 − γ)2 (cid:107)∇fik (wk)(cid:107)2
To bound the inner-product, we use Individual Convexity to relate it to the optimality gap,

(cid:104)∇fik (wk), wk + mk − w∗(cid:105) +

−

A−1
k

2ηk
1 − γ

η2
k

(15)

.

γ
(cid:104)∇fik (wk), wk + mk − w∗(cid:105) = (cid:104)∇fik (wk), wk − w∗(cid:105) +
1 − γ
γ
1 − γ

≥ fik (wk) − fik (w∗) +

(cid:104)∇fik (wk), wk − wk−1(cid:105),

[fik (wk) − fik (wk−1)],

=

1
1 − γ

[fik (wk) − fik (w∗)] −

γ
1 − γ

[fik (wk−1) − fik (w∗)].

34

Under review as a conference paper at ICLR 2021

To bound the gradient norm, we use the step-size assumption that

ηk (cid:107)∇fik (wk)(cid:107)2

A−1
k

≤ M [fik (wk) − f ∗
ik

] = M [fik (wk) − fik (w∗)] + M [fik (w∗) − f ∗
ik

].

For simplicity of notation, let us deﬁne the shortcuts

hk(w) = fik (w) − fik (w∗),

k = fik (w∗) − f ∗
σ2
ik

.

Plugging those two inequalities in the recursion of Eq. (15) gives

(cid:107)δk+1(cid:107)2
Ak

≤ (cid:107)δk(cid:107)2
Ak

−

ηk

(1 − γ)2 (2 − M )hk(wk) +

2ηkγ
(1 − γ)2 hk(wk−1) +

M ηk
(1 − γ)2 σ2
k.

We can now divide by ηk/(1−γ)2 and reorganize the inequality as

(2 − M )hk(wk) − 2γhk(wk−1) ≤

(cid:16)

(1 − γ)2
ηk

(cid:107)δk(cid:107)2
Ak

− (cid:107)δk+1(cid:107)2
Ak

(cid:17)

+ M σ2
k.

Taking the average over all iterations, the inequality yields

1
T

T
(cid:88)

(2 − M )hk(wk) − 2γhk(wk−1) ≤

k=1

1
T

T
(cid:88)

k=1

(1 − γ)2
ηk

(cid:16)
(cid:107)δk(cid:107)2
Ak

− (cid:107)δk+1(cid:107)2
Ak

(cid:17)

+ M σ2
k.

To bound the right-hand side, under the assumption that the iterates are bounded by (cid:107)wk − w∗(cid:107) ≤ D,
we use Young’s inequality to get a bound on (cid:107)δk(cid:107)2;

(cid:107)δk(cid:107)2

2 = (cid:107)wk + mk − w∗(cid:107)2

2 =

≤

(cid:16)

2
(1 − γ)2

(cid:107)wk − w∗(cid:107)2

(cid:13)
(cid:13)
(cid:13)

1

1−γ (wk − w∗) − γ
(cid:17)

2 + γ2 (cid:107)wk−1 − w∗(cid:107)2

2

(cid:13)
2
(cid:13)
1−γ (wk−1 − w∗)
(cid:13)
2
2(1 + γ2)
(1 − γ)2 D2 = ∆2.

≤

Given the upper bound (cid:107)δk(cid:107)2 ≤ ∆, a reorganization of the sum lets us apply Lemma 3 to get

(cid:16)

(cid:80)T

k=1

1
ηk

(cid:107)δk(cid:107)2
Ak

− (cid:107)δk+1(cid:107)2
Ak

(cid:17)

= (cid:80)T

= (cid:80)T

≤ (cid:80)T

= (cid:80)T

k=1 (cid:107)δk(cid:107)2
k=1 (cid:107)δk(cid:107)2
k=1 (cid:107)δk(cid:107)2
k=1 (cid:107)δk(cid:107)2

1
ηk

1
ηk

1
ηk

− (cid:80)T

− (cid:80)T +1

1
ηk

k=1 (cid:107)δk+1(cid:107)2
k=2 (cid:107)δk(cid:107)2
k=1 (cid:107)δk(cid:107)2

1
ηk−1

− (cid:80)T

Ak

Ak

Ak

Ak

Ak−1

1
Ak−1
ηk−1
≤ ∆2amaxd
ηmin

,

1
ηk

Ak− 1

ηk−1

Ak−1

+ (cid:107)δ1(cid:107)2

1
η0

A0

where the last step uses the convention A0 = 0 and Lemma 3 on δk instead of wk − w∗. Plugging
this inequality in, we get the simpler bound on the right-hand-side

1
T

T
(cid:88)

(2 − M )hk(wk) − 2γhk(wk−1) ≤

k=1

2(1 + γ2)D2amaxd
T ηmin

+ M σ2
k.

Now that the step-size is bounded deterministically, we can take the expectation on both sides to get

1
T

(cid:34) T

(cid:88)

E

(2 − M )h(wk) − 2γh(wk−1)

≤

(cid:35)

k=1

2(1 + γ2)D2amaxd
T ηmin

+ M σ2,

where h(w) = f (w) − f ∗ and σ2 = E(cid:2)fik (w∗) − f ∗
the weights on the optimality gaps to get a telescoping sum,

ik

(cid:3). To simplify the left-hand-side, we change

(cid:80)T

k=1(2 − M )h(wk) − 2γh(wk−1) = (cid:80)T

k=1(2 − 2γ − M )h(wk) + 2γh(wk) − 2γh(wk−1),
(cid:105)

+ 2γ(h(wT ) − h(w0)),

= (2 − 2γ − M )

(cid:104)(cid:80)T

(cid:104)(cid:80)T

k=1 h(wk)
(cid:105)
k=1 h(wk)

≥(2 − 2γ − M )

− 2γh(w0).

35

Under review as a conference paper at ICLR 2021

The last inequality uses h(wT ) ≥ 0. Moving the initial optimality gap to the right-hand-side, we get

1
T

(2 − 2γ − M ) E

(cid:34) T

(cid:88)

k=1

(cid:35)

h(wk)

≤

1
T

(cid:18) 2(1 + γ2)D2amaxd
ηmin

(cid:19)

+ 2γh(w0)

+ M σ2.

Assuming 2 − 2γ − M > 0 and dividing, we get

1
T

(cid:34) T

(cid:88)

E

k=1

(cid:35)

h(wk)

≤

1
2 − 2γ − M

(cid:20) 1
T

(cid:18) 2(1 + γ2)D2amaxd
ηmin

(cid:19)

+ 2γh(w0)

+ M σ2

(cid:21)
.

Using Jensen’s inequality and averaging the iterates ﬁnishes the proof.

36

Under review as a conference paper at ICLR 2021

F EXPERIMENTAL DETAILS

Our proposed adaptive gradient methods with SLS and SPS step-sizes are presented in Algorithms 1
and 3. We now make a few additional remarks on the practical use of these methods.

pk ← A−1

k ∇fik (wk)

pk ← ∇fik (wk)

(cid:46) Form the preconditioner.

if k == 0 then
ηk ← ηmax

else if mode == Armijo then

end if
if conservative then

ik ← sample mini-batch of size b
Ak ← precond(k)
if mode == Lipschitz then

Algorithm 1 Adaptive methods with SLS(f , precond, β, conservative, mode, w0, ηmax, b,
c ∈ (0, 1), γ < 1)
1: for k = 0, . . . , T − 1 do
2:
3:
4:
5:
6:
7:
8:
9:
10:
11:
12:
13:
14:
15:
16:
17:
18:
19:
20:
21:
22:
23: end for
24: return wT

end if
while fik (wk − ηk · pk) > fik (wk) − c ηk (cid:104)∇fik (wk), pk(cid:105) do

end while
mk ← βmk−1 + (1 − β)∇fik (wk)
wk+1 ← wk − ηkA−1

(cid:46) Line-search loop

ηk ← ηmax

ηk ← ηk−1

ηk ← γ ηk

k mk

end if

else

else

η ← η

Algorithm 2 reset(η, ηmax, k, b, n, γ, opt)
1: if k = 0 then
return ηmax
2:
3: else if opt= 0 then
4:
5: else if opt= 1 then
6:
7: else if opt= 2 then
η ← ηmax
8:
9: end if
10: return η

η ← η · γb/n

As suggested by Vaswani et al. (2019b), the standard backtracking search can sometimes result in
step-sizes that are too small while taking bigger steps can yield faster convergence. To this end, we
adopted their strategies to reset the initial step-size at every iteration (Algorithm 2). In particular,
using reset option 0 corresponds to starting every backtracking line search from the step-size used
in the previous iteration. Since the backtracking never increases the step-size, this option enables
the “conservative step-size“ constraint for the Lipschitz line-search to be automatically satisﬁed.
For the Armijo line-search, we use the heuristic from Vaswani et al. (2019b) corresponding to reset
option 1. This option begins every backtracking with a slightly larger (by a factor of γ b/n, γ = 2
throughout our experiments) step-size compared to the step-size at the previous iteration, and works
well consistently across our experiments. Although we do not have theoretical guarantees for Armijo

37

Under review as a conference paper at ICLR 2021

SLS with general preconditioners such as Adam, our experimental results indicate that this is in fact
a promising combination that also performs well in practice.

i ]n

i=1, precond, β,conservative, mode, w0,

(cid:46) Form the preconditioner

pk ← A−1

k ∇fik (wk)

pk ← ∇fik (wk)

else if mode == Armijo then

ik ← sample mini-batch of size b
Ak ← precond(k)
if mode == Lipschitz then

Algorithm 3 Adaptive methods with SPS(f , [f ∗
ηmax, b, c)
1: for k = 0, . . . , T − 1 do
2:
3:
4:
5:
6:
7:
8:
9:
10:
11:
12:
13:
14:
15:
16:
17:
18:

end if
if conservative then

end if
ηk ← min
mk ← βmk−1 + (1 − β)∇fik (wk)
wk+1 ← wk − ηkA−1

c (cid:104)∇fik (wk), pk(cid:105) , ηB

(cid:110) fik (wk)−f ∗
ik

if k == 0 then

ηB ← ηmax

ηB ← ηmax

ηB ← ηk−1

end if

else

else

(cid:111)

)

k mk

19:
20:
21: end for
22: return wT

On the other hand, rather than being too conservative, the step-sizes produced by SPS between
successive iterations can vary wildly such that convergence becomes unstable. Loizou et al. (2020)
suggested to use a smoothing procedure that limits the growth of the SPS from the previous iteration
to the current. We use this strategy in our experiments with τ = 2b/n and show that both SPS and
Armijo SPS work well. For the convex experiments, for both SLS and SPS, we set c = 0.5 as is
suggested by the theory. For the non-convex experiments, we observe that all values of c ∈ [0.1, 0.5]
result in reasonably good performance, but use the values suggested in Vaswani et al. (2019b); Loizou
et al. (2020), i.e. c = 0.1 for all adaptive methods using SLS and c = 0.2 for methods using SPS.

38

Under review as a conference paper at ICLR 2021

G ADDITIONAL EXPERIMENTAL RESULTS

In this section, we present additional experimental results showing the effect of the step-size for
adaptive gradient methods using a synthetic dataset (Fig. 4). We show the wall-clock times for
the optimization methods (Fig. 5). We show the variation in the step-size for the SLS methods
when training deep networks for both the CIFAR in Fig. 6 and ImageNet (Fig. 7) datasets. We
evaluate these methods on easy non-convex objectives - classiﬁcation on MNIST (Fig. 8) and
deep matrix factorization (Fig. 10). We use deep matrix factorization to examine the effect of
over-parameterization on the performance of the optimization methods and check the methods’
performance when minimizing convex objectives associated with binary classiﬁcation using RBF
kernels in Fig. 9. Finally in Fig. 11, we quantify the gains of incorporating momentum in AMSGrad
by comparing against the performance AMSGrad without momentum.

(a) AdaGrad

(b) AMSGrad

Figure 4: Effect of step-size on the performance of adaptive gradient methods for binary classiﬁcation
on a linearly separable synthetic dataset with different margins. We observe that the large variance
for the adaptive gradient methods, and the variants with SLS have consistently good performance
across margins and optimizers.

39

050100150200Epoch102101100101Train loss (log)Margin:0.01050100150200Epoch102101100101Margin:0.05050100150200Epoch108106104102100Margin:0.1050100150200Epoch107105103101Margin:0.5AdagradDefault AdagradAdagrad + Lipschitz LSAdagrad + Armijo LS050100150200Epoch102101100101Train loss (log)Margin:0.01050100150200Epoch107105103101101Margin:0.05050100150200Epoch108106104102100Margin:0.1050100150200Epoch105103101Margin:0.5AmsgradDefault AmsgradAmsgrad + SLSUnder review as a conference paper at ICLR 2021

(a)

(b)

(c)

(d)

Figure 5: Runtime (in seconds/epoch) for optimization methods for multi-class classiﬁcation using
the deep network models in Fig. 2. Although the runtime/epoch is larger for the SLS/SPS variants,
they require fewer epochs to reach the maximum test accuracy (Figure 2). This justiﬁes the moderate
increase in wall-clock time.

40

Methods020406080100Average training time/epoch77.03875.912113.39777.01262.96263.20391.689CIFAR10 - ResNet34Amsgrad + SLSAmsgrad +  SLS + HBAdagrad +  SLSAdaboundRadamAdamSLSMethods050100150200250300Average training time/epoch289.624203.612221.197114.14698.33987.487108.076CIFAR100 - DenseNet121Amsgrad + SLSAmsgrad +  SLS + HBAdagrad +  SLSAdaboundRadamAdamSLSMethods020406080100Average training time/epoch78.19575.18881.05040.32931.36826.686103.426CIFAR100 - ResNet34Amsgrad + SLSAmsgrad +  SLS + HBAdagrad +  SLSAdaboundRadamAdamSLSMethods0255075100125150175Average training time/epoch174.929154.330126.805100.10082.629115.225111.041Tiny ImageNet - ResNet18Amsgrad + SLSAmsgrad +  SLS + HBAdaboundRadamAdamSLSAdagrad +  SLSUnder review as a conference paper at ICLR 2021

(a) CIFAR-10 ResNet

(b) CIFAR-10 DenseNet

(c) CIFAR-100 ResNet

(d) CIFAR-100 DenseNet

Figure 6: Comparing optimization methods on image classiﬁcation tasks using ResNet and DenseNet
models on the CIFAR-10/100 datasets. For the SLS/SPS variants, refer to the experimental details
in Appendix F. For Adam, we did a grid-search and use the best step-size. We use the default
hyper-parameters for the other baselines. We observe the consistently good performance of AdaGrad
and AMSGrad with Armijo SLS. We also show the variation in the step-size and observe a cyclic
pattern (Loshchilov & Hutter, 2017) - an initial warmup in the learning rate followed by a decrease or
saturation to a small step-size (Goyal et al., 2017).

41

050100150200Epoch103102101100Train loss (log)CIFAR10-ResNet34050100150200Epoch0.860.880.900.920.94Validation accuracyCIFAR10-ResNet34050100150200Epoch10121010108106104102100Step size (log)CIFAR10-ResNet34Adagrad +  SLSAdaboundRadamAdamSLSAmsgrad + SLSAmsgrad +  SLS + HB050100150200Epoch103102101100Train loss (log)CIFAR10-DenseNet121050100150200Epoch0.860.880.900.920.94Validation accuracyCIFAR10-DenseNet121050100150200Epoch1012109106103100Step size (log)CIFAR10-DenseNet121Adagrad +  SLSAdaboundRadamAdamSLSAmsgrad + SLSAmsgrad +  SLS + HB050100150200Epoch103102101100Train loss (log)CIFAR100-ResNet34050100150200Epoch0.660.680.700.720.740.76Validation accuracyCIFAR100-ResNet34050100150200Epoch105103101101Step size (log)CIFAR100-ResNet34Adagrad +  SLSAdaboundRadamAdamSLSAmsgrad + SLSAmsgrad +  SLS + HB50100150200Epoch103102101100Train loss (log)CIFAR100-DenseNet12150100150200Epoch0.660.680.700.720.740.76Validation accuracyCIFAR100-DenseNet12150100150200Epoch108106104102100102Step size (log)CIFAR100-DenseNet121Adagrad +  SLSAdaboundRadamAdamSLSAmsgrad + SLSAmsgrad +  SLS + HBUnder review as a conference paper at ICLR 2021

(a) Imagewoof

(b) ImageNette

(c) Tiny Imagenet

Figure 7: Comparing optimization methods on image classiﬁcation tasks using variants of ImageNet.
We use the same settings as the CIFAR datasets and observe that AdaGrad and AMSGrad with Armijo
SLS is consistently better.

Figure 8: Comparing optimization methods on MNIST.

42

20406080100Epoch106105104103102101100Train loss (log)Imagewoof-ResNet1820406080100Epoch0.5000.5250.5500.5750.6000.6250.6500.6750.700Validation accuracyImagewoof-ResNet1820406080100Epoch105103101101Step size (log)Imagewoof-ResNet18AdaboundRadamAdamSLSAmsgrad + SLSAdagrad +  SLSAmsgrad +  SLS + HB020406080100Epoch106105104103102101100Train loss (log)Imagenette-ResNet18020406080100Epoch0.760.770.780.790.800.810.820.830.84Validation accuracyImagenette-ResNet18020406080100Epoch108106104102100Step size (log)Imagenette-ResNet18AdaboundRadamAdamSLSAmsgrad + SLSAdagrad +  SLSAmsgrad +  SLS + HB5075100125150175200Epoch103102101100Train loss (log)Tiny ImageNet-ResNet18050100150200Epoch0.340.350.360.370.380.390.40Validation accuracyTiny ImageNet-ResNet18050100150200Epoch105103101101Step size (log)Tiny ImageNet-ResNet18AdaboundRadamAdamSLSAmsgrad + SLSAdagrad +  SLSAmsgrad +  SLS + HB020406080100Epoch104103102101100Train loss (log)MNIST020406080100Epoch0.9760.9780.9800.9820.984Validation accuracyMNIST020406080100Epoch105104103102101100Step size (log)MNISTAdamAdaboundRadamSLSAdagrad +  SLSAmsgrad + SLSAmsgrad +  SLS + HBUnder review as a conference paper at ICLR 2021

Figure 9: Comparison of optimization methods on convex objectives: binary classiﬁcation on LIBSVM
datasets using RBF kernel mappings. The kernel bandwidths are chosen by cross-validation following
the protocol in (Vaswani et al., 2019b). All line-search methods use c = 1/2 and the procedure
described in Appendix F. The other methods are use their default parameters. We observe the superior
convergence of the SLS variants and the poor performance of the baselines.

Figure 10: Comparison of optimization methods for deep matrix factorization. Methods use the
same hyper-parameter settings as above and we examine the effects of over-parameterization on the
Ex∼N (0,I) (cid:107)W2W1x − Ax(cid:107)2 (Vaswani et al., 2019b; Rolinek & Martius, 2018).
problem: minW1,W2
We choose A ∈ R10×6 with condition number κ(A) = 1010 and control the over-parameterization
via the rank k (equal to 1,4, 10) of W1 ∈ Rk×6 and W2 ∈ R10×k. We also compare against the
true model. In each case, we use a ﬁxed dataset of 1000 samples. We observe that as the over-
parameterization increases, the performance of all methods improves, with the methods equipped
with SLS performing the best.

43

020406080100Epoch103102101Train loss (log)ijcnn020406080100Epoch109107105103101mushrooms020406080100Epoch102101rcv1AdaboundRadamAdamSLSAdagrad +  SLSAmsgrad + SLSAmsgrad +  SLS + HB0255075100Epoch10131010107104101Train loss (log)True model0255075100Epoch2×1013×1014×1016×101Rank 10255075100Epoch10131010107104101Rank 100255075100Epoch102101Rank 4AdamAdaboundRadamSLSAdagrad +  SLSAmsgrad + SLSAmsgrad +  SLS + HBUnder review as a conference paper at ICLR 2021

Figure 11: Ablation study comparing variants of the basic optimizers for multi-class classiﬁcation
with deep networks. Training loss (top) and validation accuracy (bottom) for CIFAR-10, CIFAR-
100 and Tiny ImageNet. We consider the AdaGrad with AMSGrad-like momentum and do not ﬁnd
improvements in performance. We also benchmark the performance of AMSGrad without momentum,
and observe that incorporating AMSGrad momentum does improve the performance, whereas heavy-
ball momentum has a minor, sometimes detrimental effect. We use SLS and Adam as benchmarks to
study the effects of incorporating preconditioning vs step-size adaptation.

44

050100150200Epoch104103102101100Train loss (log)CIFAR10 - ResNet3450100150200Epoch103102101100CIFAR100 - DenseNet121050100150200Epoch103102101100CIFAR100 - ResNet34050100150200Epoch103102101100101Tiny ImageNet - ResNet18050100150200Epoch0.860.880.900.920.94Validation accuracyCIFAR10 - ResNet3450100150200Epoch0.660.680.700.720.740.76CIFAR100 - DenseNet121050100150200Epoch0.660.680.700.720.740.76CIFAR100 - ResNet3450100150200Epoch0.340.350.360.370.380.390.40Tiny ImageNet - ResNet18Amsgrad + SLSAdagrad + SLS + momAmsgrad +  SLS + HBAdamSLSAmsgrad +  SLS (beta = 0)Adagrad +  SLS