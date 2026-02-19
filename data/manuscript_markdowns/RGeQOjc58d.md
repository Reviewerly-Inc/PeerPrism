Under review as a conference paper at ICLR 2021

IMPROVED GRADIENT BASED ADVERSARIAL
ATTACKS FOR QUANTIZED NETWORKS

Anonymous authors
Paper under double-blind review

ABSTRACT

Neural network quantization has become increasingly popular due to efﬁcient
memory consumption and faster computation resulting from bitwise operations
on the quantized networks. Even though they exhibit excellent generalization
capabilities, their robustness properties are not well-understood. In this work,
we systematically study the robustness of quantized networks against gradient
based adversarial attacks and demonstrate that these quantized models suffer from
gradient vanishing issues and show a fake sense of robustness. By attributing
gradient vanishing to poor forward-backward signal propagation in the trained
network, we introduce a simple temperature scaling approach to mitigate this issue
while preserving the decision boundary. Despite being a simple modiﬁcation to
existing gradient based adversarial attacks, experiments on CIFAR-10/100 datasets
with multiple network architectures demonstrate that our temperature scaled attacks
obtain near-perfect success rate on quantized networks while outperforming original
attacks on adversarially trained models as well as ﬂoating-point networks.

1

INTRODUCTION

Neural Network (NN) quantization has become increasingly popular due to reduced memory and
time complexity enabling real-time applications and inference on resource-limited devices. Such
quantized networks often exhibit excellent generalization capabilities despite having low capacity due
to reduced precision for parameters and activations. However, their robustness properties are not well-
understood. In particular, while parameter quantized networks are claimed to have better robustness
against gradient based adversarial attacks (Galloway et al. (2018)), activation only quantized methods
are shown to be vulnerable (Lin et al. (2019)).

In this work, we consider the extreme case of Binary Neural Networks (BNNs) and systematically
study the robustness properties of parameter quantized models, as well as both parameter and
activation quantized models against gradient based adversarial attacks. Our analysis reveals that these
quantized models suffer from gradient masking issues (Athalye et al. (2018)) (especially vanishing
gradients) and in turn show fake robustness. We attribute this vanishing gradients issue to poor
forward-backward signal propagation caused by trained binary weights, and our idea is to improve
signal propagation of the network without affecting the prediction of the classiﬁer.

There is a body of work on improving signal propagation in a neural network (e.g., Glorot & Bengio
(2010); Pennington et al. (2017); Lu et al. (2020)), however, we are facing a unique challenge of
improving signal propagation while preserving the decision boundary, since our ultimate objective
is to generate adversarial attacks. To this end, we ﬁrst discuss the conditions to ensure informative
gradients and then resort to a temperature scaling approach (Guo et al. (2017)) (which scales the
logits before applying softmax cross-entropy) to show that, even with a single positive scalar the
vanishing gradients issue in BNNs can be alleviated achieving near perfect success rate in all tested
cases.

Speciﬁcally, we introduce two techniques to choose the temperature scale: 1) based on the singular
values of the input-output Jacobian, 2) by maximizing the norm of the Hessian of the loss with
respect to the input. The justiﬁcation for the ﬁrst case is that if the singular values of input-output
Jacobian are concentrated around 1 (deﬁned as dynamical isometry (Pennington et al. (2017))) then
the network is said to have good signal propagation and we intend to make the mean of singular

1

Under review as a conference paper at ICLR 2021

values to be 1. On the other hand, the intuition for maximizing the Hessian norm is that if the Hessian
norm is large, then the gradient of the loss with respect to the input is sensitive to an inﬁnitesimal
change in the input. This is a sufﬁcient condition for the network to have good signal propagation as
well as informative gradients under the assumption that the network does not have any randomized or
non-differentiable components.

We evaluated our improved gradient based adversarial attacks using BNNs with weight quantized
(BNN-WQ) and weight and activation quantized (BNN-WAQ), ﬂoating point networks (REF), and
adversarially trained models. We employ quantized and ﬂoating point networks trained on CIFAR-
10/100 datasets using several architectures. In all tested BNNs, both versions of our temperature scaled
attacks obtained near-perfect success rate outperforming gradient based attacks (FGSM (Goodfellow
et al. (2014)), PGD (Madry et al. (2017))). Furthermore, this temperature scaling improved gradient
based attacks even on adversarially trained models (both high-precision and quantized) as well as
ﬂoating point networks, showing the signiﬁcance of signal propagation for adversarial attacks.

2 PRELIMINARIES

We ﬁrst provide some background on the neural network quantization and adversarial attacks.

2.1 NEURAL NETWORK QUANTIZATION

Neural Network (NN) quantization is deﬁned as training networks with parameters constrained to a
minimal, discrete set of quantization levels. This primarily relies on the hypothesis that since NNs are
usually overparametrized, it is possible to obtain a quantized network with performance comparable
to the ﬂoating point network. Given a dataset D = {xi, yi}n
i=1, NN quantization can be written as:

1
n
Here, (cid:96)(·) denotes the input-output mapping composed with a standard loss function (e.g., cross-
entropy loss), w is the m dimensional parameter vector, and Q is a predeﬁned discrete set representing
quantization levels (e.g., Q = {−1, 1} in the binary case).

(cid:96)(w; (xi, yi)) .

L(w; D) :=

min
w∈Qm

(1)

i=1

n
(cid:88)

Most of the NN quantization approaches (Ajanthan et al. (2019a;b); Bai et al. (2019); Hubara et al.
(2017)) convert the above problem into an unconstrained problem by introducing auxiliary variables
and optimize via (stochastic) gradient descent. To this end, the algorithms differ in the choice of
quantization set (e.g., keep it discrete (Courbariaux et al. (2015)), relax it to the convex hull (Bai
et al. (2019)) or convert the problem into a lifted probability space (Ajanthan et al. (2019a))), the
projection used, and how differentiation through projection is performed. In the case when the
constraint set is relaxed, a gradually increasing annealing hyperparameter is used to enforce a
quantized solution (Ajanthan et al. (2019a;b); Bai et al. (2019)). We refer the interested reader to
respective papers for more detail. In this paper, we use BNN-WQ obtained using MD-tanh-S (Ajanthan
et al. (2019b)) and BNN-WAQ obtained using Hubara et al. (2017).

2.2 ADVERSARIAL ATTACKS

Adversarial examples consist of imperceptible perturbations to the data that alter the model’s predic-
tion with high conﬁdence. Existing attacks can be categorized into white-box and black-box attacks
where the difference lies in the knowledge of the adversaries. White-box attacks allow the adversaries
access to the target model’s architecture and parameters, whereas black-box attacks can only query
the model. Since white-box gradient based attacks are popular, we summarize them below.

First-order gradient based attacks can be compactly written as Projected Gradient Descent (PGD) on
the negative of the loss function (Madry et al. (2017)). Formally, let x0 ∈ IRN be the input image,
then at iteration t, the PGD update can be written as:

xt+1 = P (cid:0)xt + η gt

x

(cid:1) ,

(2)

where P : IRN → X is a projection, X ⊂ IRN is the constraint set that bounds the perturbations,
η > 0 is the step size, and gt
x is a form of gradient of the loss with respect to the input x evaluated at
xt. With this general form, the popular gradient based adversarial attacks can be speciﬁed:

• Fast Gradient Sign Method (FGSM): This is a one step attack introduced in Goodfellow et al.
(2014). Here, P is the identity mapping, η is the maximum allowed perturbation magnitude, and

2

Under review as a conference paper at ICLR 2021

Method

ResNet-18

VGG-16

Clean Adv.(1) Adv.(20) Clean Adv.(1) Adv.(20)

REF
BNN-WQ
BNN-WAQ

94.46
93.18
87.67

0.00
26.98
8.57

0.00
17.91
1.94

93.31
91.53
89.69

0.04
47.32
78.01

0.00
38.49
59.26

Table 1: Clean and adversarial accuracy (PGD attack with L∞ bound) on the test set of CIFAR-10
using ResNet-18 and VGG-16. In brackets, we mention number of random restarts used to perform the
attack. Note, BNNs outperform adversarial accuracy of ﬂoating point networks consistently.

(a)

(b)
Figure 1: Gradient marking checks in ResNet-18 on CIFAR-10 for PGD attack with L∞ bound: (a)
varying iterations, (b) varying radius, and (c) black-box attacks on ResNet-18 and VGG-16. While (a),
(c) show signs of gradient masking, (b) does not. We attribute this discrepancy to the random initial
step before PGD.

(c)

x = sign (∇x(cid:96)(w∗; (xt, y))), where (cid:96) denotes the loss function, w∗ is the trained weights and y
gt
is the ground truth label corresponding to the image x0.

• PGD with L∞ bound: Arguably the most popular adversarial attack introduced in Madry et al.
(2017) and sometimes referred to as Iterative Fast Gradient Sign Method (IFGSM). Here, P is the
x = sign (∇x(cid:96)(w∗; (xt, y))), the sign of
L∞ norm based projection, η is a chosen step size, and gt
gradient same as FGSM.

• PGD with L2 bound: This is also introduced in Madry et al. (2017) which performs the standard
PGD in the Euclidean space. Here, P is the L2 norm based projection, η is a chosen step size, and
x = ∇x(cid:96)(w∗; (xt, y)) is simply the gradient of the loss with respect to the input.
gt

These attacks have been further strengthened by a random initial step (Tramèr et al. (2017)). In this
paper, we perform this single random initialization for all experiments with FGSM/PGD attack unless
otherwise mentioned.

3 ROBUSTNESS EVALUATION OF BINARY NEURAL NETWORKS

We start by evaluating the adversarial accuracy (i.e. accuracy on the perturbed data) of BNNs using
the PGD attack with L∞ bound.

• PGD attack details: perturbation bound of 8 pixels (assuming each pixel in the image is in [0, 255])
with respect to L∞ norm, step size η = 2 and the total number of iterations T = 20. The attack
details are the same in all evaluated settings unless stated otherwise.

We perform experiments on CIFAR-10 dataset using ResNet-18 and VGG-16 architectures and report
the clean accuracy and PGD adversarial accuracy with 1 and 20 random restarts in Table 1. It can
be clearly and consistently observed that binary networks have high adversarial accuracy compared
to the ﬂoating point counterparts. Even with 20 random restarts, BNNs clearly outperform ﬂoating
point networks in terms of adversarial accuracy. Since this result is surprising, we investigate this
phenomenon further to understand whether BNNs are actually robust to adversarial perturbations or
they show a fake sense of security due to some form of obfuscated gradients (Athalye et al. (2018)).

3.1

IDENTIFYING OBFUSCATED GRADIENTS

Recently, it has been shown that several defense mechanisms intentionally or unintentionally break
gradient descent and cause obfuscated gradients and thus exhibit a false sense of security (Athalye et al.
(2018)). Several gradient based adversarial attacks tend to fail to produce adversarial perturbations in
scenarios where the gradients are uninformative, referred to as gradient masking. Gradient masking

3

01020304050Attack Iterations020406080100Adversarial accuracy (in %)Varying PGD Attack InterationsREFBNN­WQBNN­WAQ01020304050607080Attack Radius010203040506070Adversarial accuracy (in %)Varying PGD Attack RadiusREFBNN­WQBNN­WAQBNN-WQ(ResNet-18)BNN-WQ(VGG-16)BNN-WAQ(ResNet-18)BNN-WAQ(VGG-16)01020304050607080Adversarial accuracy (in %)White Box Acc.Black Box Acc.Under review as a conference paper at ICLR 2021

can occur due to shattered gradients, stochastic gradients or exploding and vanishing gradients. We
try to identify gradient masking in binary networks based on the empirical checks provided in Athalye
et al. (2018). If any of these checks fail, it indicates gradient masking issue in BNNs.

To illustrate this, we analyse the effects of varying different hyperparameters of PGD attack on BNNs
trained on CIFAR-10 using ResNet-18 architecture. Even though varying PGD perturbation bound
does not show any signs of gradient masking, varying attack iterations and black-box vs white-box
results (on ResNet-18 and VGG-16) clearly indicate gradient masking issues as depicted in Fig. 1.
The black-box attack outperforming white-box attack for BNNs certainly indicates gradient masking
issues since the black-box attack do not use the gradient information from model being attacked.
Here, our black-box model to a BNN is the analogous ﬂoating point network trained on the same
dataset and the attack is the same PGD with L∞ bound.

These checks demonstrate that BNNs are prone to gradient masking and exhibit fake robustness.
Note, shattered gradients occur due to non-differentiable components in the defense mechanism and
stochastic gradients are caused by randomized gradients. Since BNNs are trainable from scratch
and does not have randomized gradients1, we narrow down gradient masking issue to vanishing or
exploding gradients. Since, vanishing or exploding gradients occur due to poor signal propagation,
by introducing a single scalar, we discuss two approaches to mitigate this issue, which lead to almost
100% success rate for gradient based attacks on BNNs.

4 SIGNAL PROPAGATION OF NEURAL NETWORKS

We ﬁrst describe how poor signal propagation in neural networks can cause vanishing or exploding
gradients. Then we discuss the idea of introducing a single scalar to improve the existing gradient
based attacks without affecting the prediction (i.e., decision boundary) of the trained models.
We consider a neural network fw for an input x0, having logits aK = fw(x0). Now, since softmax
cross-entropy is usually used as the loss function, we can write:

(cid:96)(aK, y) = −yT log(p) ,

p = softmax(aK) ,

(3)

where y ∈ IRd is the one-hot encoded target label and log is applied elementwise.

For various gradient based adversarial attacks discussed in Sec. 2.2, gradient of the loss (cid:96) is used with
respect to the input x0, which can also be formulated using chain rule as,

∂(cid:96)(aK, y)
∂x0

=

∂(cid:96)(aK, y)
∂aK

∂aK
∂x0 = ψ(aK, y) J ,

(4)

where ψ denotes the error signal and J ∈ Rd×N is the input-output Jacobian. Here we use the
convention that ∂v/∂u is of the form v-size × u-size.

Notice there are two components that inﬂuence the gradients, 1) the Jacobian J and 2) the error signal
ψ. Gradient based attacks would fail if either the Jacobian is poorly conditioned or the error signal
has saturating gradients, both of these will lead to vanishing gradients in ∂(cid:96)/∂x0.

The effects of Jacobian on the signal propagation is studied in dynamical isometry and mean-ﬁeld
theory literature (Pennington et al. (2017); Saxe et al. (2013)) and it is known that a network is said to
satisfy dynamical isometry if the singular values of J are concentrated near 1. Under this condition,
error signals ψ backpropagate isometrically through the network, approximately preserving its norm
and all angles between error vectors. Thus, as dynamical isometry improves the trainability of the
ﬂoating point networks, a similar technique can be useful for gradient based attacks as well.

In fact, almost all initialization techniques (e.g., Glorot & Bengio (2010)) approximately ensures
that the Jacobian J is well-conditioned for better trainability and it is hypothesized that approximate
isometry is preserved even at the end of the training. But, for BNNs, the weights are constrained to be
{−1, 1} and hence the weight distribution at end of training is completely different from the random
initialization. Furthermore, it is not clear that fully-quantized networks can achieve well-conditioned
Jacobian, which guided some research activity in utilizing layerwise scalars (either predeﬁned or
learned) to improve BNN training (McDonnell (2018); Rastegari et al. (2016)). We would like to
point out that the focus of this paper is to improve gradient based attacks on already trained BNNs. To

1 BNN-WQ have binary weights, but there is no non-differentiable or randomized component once trained.

4

Under review as a conference paper at ICLR 2021

this end learning a new scalar to improve signal propagation at each layer is not useful as it can alter
the decision boundary of the network and thus cannot be used in practice on already trained model.

4.1 TEMPERATURE SCALING FOR BETTER SIGNAL PROPAGATION

In this paper, we propose to use a single scalar per network to improve the signal propagation of the
network using temperature scaling. In fact, one could replace softmax with a monotonic function
such that the prediction is not altered, however, we will show in our experiments that a single scalar
with softmax has enough ﬂexibility to improve signal propagation and yields almost 100% success
rate with PGD attacks. Essentially, we can use a scalar, β > 0 without changing the decision boundary
of the network by preserving the relative order of the logits. Precisely, we consider the following:

p(β) = softmax(¯aK) ,

¯aK = β aK .

(5)

Here, we write the softmax output probabilities p as a function of β to emphasize that they are softmax
output of temperature scaled logits. Now since in this context, the only variable is the temperature
scale β, we denote the loss and the error signal as functions of only β. With this simpliﬁed notation
the gradient of the temperature scaled loss with respect to the inputs can be written as:

∂(cid:96)(β)
∂x0 =

∂(cid:96)(β)
∂¯aK

∂¯aK
∂aK

∂aK
∂x0 = ψ(β)β J .

(6)

Note that β affects the input-output Jacobian linearly while it nonlinearly affects the error signal ψ.
To this end, we hope to obtain a β that ensures the error signal is useful (i.e., not all zero) as well as
the Jacobian is well-conditioned to allow the error signal to propagate to the input.

We acknowledge that while one can ﬁnd a β > 0 to obtain softmax output ranging from a uniform
distribution (β = 0) to one-hot vectors (β → ∞), β only scales the Jacobian. Therefore, if the
Jacobian J has zero singular values, our approach has no effect in those dimensions. However,
since most of the modern networks consist of ReLU nonlinearities (generally positive homogeneous
functions), the effect of a single scalar would be equivalent (ignoring the biases) to having layerwise
scalars such as in McDonnell (2018). Thus, we believe a single scalar is sufﬁcient for our purpose.

5

IMPROVED GRADIENTS FOR ADVERSARIAL ATTACKS

Now we discuss strategies to choose a scalar β such that
the gradients with respect to input are informative. Let us
ﬁrst analyze the effect of β on the error signal. To this end,

ψ(β) =

∂(cid:96)(β)
∂p(β)

∂p(β)
∂¯aK = −(y − p(β))T .

(7)

where y is the one-hot encoded target label, and p(β) is
the softmax output of scaled logits.

Figure 2: Error signal (ψ(β)) and Ja-
For adversarial attacks, we only consider the correctly clas-
cobian of softmax (∂p(β)/∂¯aK) vs. β
siﬁed images (i.e., argmaxj yj = argmaxj pj(β)) as there
for a random correctly classiﬁed logits.
is no need to generate adversarial examples corresponding
to misclassiﬁed samples. From the above formula, it is clear that when p(β) is one-hot encoding
then the error signal is 0. This is one of the reason for vanishing gradient issue in BNNs. Even if this
does not happen for a given image, one can increase β → ∞ to make this error signal 0. Similarly,
when p(β) is the uniform distribution, the norm of the error signal is at the maximum. This can be
obtained by setting β = 0. However, this would also make ∂(cid:96)(β)/∂x0 = 0 as the singular values of
the input-output Jacobian would all be 0. How error signal is affected by β is illustrated in Fig. 2.

This analysis indicates that the optimal β cannot be obtained by simply maximizing the norm of the
error signal and we need to balance both the Jacobian as well as the error signal. To summarize, the
scalar β should be chosen such that the following properties are satisﬁed:
1. (cid:107)ψ(β)(cid:107)2 > ρ for some ρ > 0.
2. The Jacobian β J is well-conditioned, i.e., the singular values of β J is concentrated around 1.

5.1 NETWORK JACOBIAN SCALING (NJS)

We now discuss a straightforward, two-step approach to attain the aforementioned properties. Firstly,
to ensure βJ is well-conditioned, we simply choose β to be the inverse of the mean of singular values

5

0204060801000.000.020.040.060.08JSV (Mean)||||*0.1Under review as a conference paper at ICLR 2021

of J. This guarantees that the mean of singular values of βJ is 1. After this scaling, it is possible
that the resulting error signal is very small. To ensure that (cid:107)ψ(β)(cid:107)2 > ρ > 0, we ensure that the
softmax output pk(β) corresponding to the ground truth class k is at least ρ away from 1. We now
state it as a proposition to derive β given a lowerbound on 1 − pk(β).
d and aK
Proposition 1. Let aK ∈ IRd with d > 1 and aK
For a given 0 < ρ < (d − 1)/d, there exists a β > 0 such that 1 − softmax(βaK
β < − log(ρ/(d − 1)(1 − ρ))/γ.
Proof. This is derived via a simple algebraic manipulation of softmax. Please refer to Appendix.
This β can be used together with the one computed using inverse of mean Jacobian Singular Values
(JSV). We provide the pseudocode for our proposed PGD++ (NJS) attack in Appendix. Similar
approach can also be applied for FGSM++. Notice that, this approach is simple and it adds negligible
overhead to the standard PGD attacks. However, it has a hyperparameter ρ which is hand designed.
To mitigate this, next we discuss a hyperparameter-free approach to obtain β.

1 − aK
d = γ.
1 ) > ρ, then

2 ≥ . . . ≥ aK

1 ≥ aK

5.2 HESSIAN NORM SCALING (HNS)

We now discuss another approach to obtain informative gradients. Our idea is to maximize the
Frobenius norm of the Hessian of the loss with respect to the input, where the intuition is that if
the Hessian norm is large, then the gradient ∂(cid:96)/∂x0 is sensitive to an inﬁnitesimal change in x0.
This means, the inﬁnitesimal perturbation in the input is propagated in the forward pass to the last
layer and propagated back to the input layer without attenuation (i.e., the returned signal is not zero),
assuming there are no randomized or non-differentiable components in the network. This clearly
indicates that the network has good signal propagation as well as the error signals are not all zero.
This objective can now be written as:

β∗ = argmax

β>0

(cid:13)
(cid:13)
(cid:13)
(cid:13)

∂2(cid:96)(β)
∂(x0)2

(cid:13)
(cid:13)
(cid:13)
(cid:13)F

= argmax

β>0

(cid:34)

β

ψ(β)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

∂J
∂x0 + β

(cid:18) ∂p(β)
∂¯aK J

(cid:19)T

J

(cid:35)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)F

(8)

.

The derivation is provided in Appendix. Note, since J does
not depend on β, J and ∂J/∂x0 are computed only once,
β is optimized using grid search as it involves only a single
scalar. In fact, it is easy to see from the above equation that,
when the Hessian is maximized, β cannot be zero. Simi-
larly, ψ(β) cannot be zero because if it is zero, then the pre-
diction p(β) is one-hot encoding (Eq. (7)), consequently
∂p(β)/∂¯aK = 0 and this cannot be a maximum for the
Hessian norm. Hence, this ensures that (cid:107)ψ(β∗)(cid:107)2 > ρ for
some ρ > 0 and β∗ is bounded according to Proposition 1.
Therefore, the maximum is obtained for a ﬁnite value of
β. Even though, it is not clear how exactly this approach
would affect the singular values of the input-output Jacobian (β J), we know that they are ﬁnite and
not zero. How Hessian norm is inﬂuenced by β is illustrated in Fig. 3.

Figure 3: Hessian norm vs. β on a ran-
dom correctly classiﬁed image. The plot
clearly shows a concave behaviour. s

Furthermore, there are some recent works (Moosavi-Dezfooli et al. (2019); Qin et al. (2019)) show
that adversarial training makes the loss surface locally linear around the vicinity of training samples
and enforcing local linearity constraint on loss curvature can achieve better robust to adversarial
attacks. On the contrary, our idea of maximizing the Hessian, i.e., increasing the nonlinearity of
(cid:96), could make the network more prone to adversarial attacks and we intend to exploit that. The
psuedocode for PGD++ attack with HNS is summarized in Appendix.

6 EXPERIMENTS

We evaluate robustness accuracies of BNNs with weight quantized (BNN-WQ), weight and activation
quantized (BNN-WAQ) ﬂoating point networks (REF), and adversarially trained networks. We evaluate
our two PGD++ variants corresponding to Hessian Norm Scaling (HNS) and Network Jacobian
Scaling (NJS) on CIFAR-10 and CIFAR-100 datasets with multiple network architectures. Brieﬂy,
our results indicate that both of our proposed attack variants yield attack success rate much higher
than original PGD attacks not only on L∞ bounded attack but also on L2 bounded attacks on both
ﬂoating point networks and binarized networks. Our proposed PGD++ variants also reduce PGD

6

0.00.10.20.30.40.50.000.250.500.751.001.251.501.752.00Hessian NormUnder review as a conference paper at ICLR 2021

Network

0 ResNet-18

1
-
R
A
F
I
C

VGG-16
ResNet-50
DenseNet-121
MobileNet-V2

0 ResNet-18
0
1
-
R
A
F
I
C

VGG-16
ResNet-50
DenseNet-121
MobileNet-V2

FGSM

40.49
57.55
57.62
26.80
33.50

25.22
19.82
37.76
28.32
12.09

FGSM++

NJS

3.46
4.00
6.44
4.67
6.42

14.08
7.98
16.33
12.21
10.18

HNS

2.51
3.43
5.35
4.24
5.42

1.80
1.76
14.17
10.86
8.79

Adversarial Accuracy (%)

PGD (L∞)

PGD++ (L∞)
HNS
NJS

PGD (L2)

PGD++ (L2)
HNS
NJS

26.98
47.32
43.14
9.11
26.86

8.23
17.44
25.71
8.87
1.44

0.00
0.00
0.00
0.00
0.00

2.45
0.88
2.33
1.15
0.57

0.00
0.00
0.00
0.00
0.00

0.00
0.16
2.73
1.09
0.66

74.59
61.90
74.75
72.99
35.22

42.67
19.26
38.95
43.78
8.97

0.05
0.35
0.11
0.03
0.12

6.79
3.17
7.9
4.54
3.39

0.05
1.32
0.08
0.06
0.09

0.26
0.63
7.41
4.16
3.01

Table 2: Adversarial accuracy on the test set for BNN-WQ. Both our NJS and HNS variants consistently
outperform original L∞ bounded FGSM and PGD attack, and L2 bounded PGD attack.

Network

F
E
R

ResNet-18
VGG-16
ResNet-50
DenseNet-121

Q ResNet-18
A
W

VGG-16
ResNet-50
DenseNet-121

-
N
N
B

FGSM

7.62
11.01
21.64
11.40

40.84
79.92
33.16
37.20

FGSM++

NJS

5.55
10.04
6.08
7.58

19.46
15.96
25.89
23.89

HNS

5.35
9.66
5.70
7.30

19.09
15.39
27.05
24.69

Adversarial Accuracy (%)

PGD (L∞)

PGD++ (L∞)
HNS
NJS

PGD (L2)

PGD++ (L2)
HNS
NJS

0.00
0.04
0.69
0.00

8.57
78.01
0.49
0.81

0.00
0.00
0.00
0.00

0.03
0.01
0.23
0.10

0.00
0.00
0.00
0.00

0.04
0.02
0.45
0.18

45.18
2.23
65.56
38.15

67.84
85.62
32.93
59.32

0.09
0.78
0.07
0.08

2.33
0.49
6.68
3.72

0.05
1.10
0.09
0.06

2.59
0.62
8.77
6.17

Table 3: Adversarial accuracy on the test set of CIFAR-10 for REF and BNN-WAQ. Both our NJS and
HNS variants consistently outperform original FGSM and PGD (L∞/L2 bounded) attacks.

adversarial accuracy of adversarially trained ﬂoating point and adversarially trained binarized neural
networks while outperforming much stronger attacks such as DeepFool (Moosavi-Dezfooli et al.
(2016)) and Brendel & Bethge Attack (BBA) (Brendel et al. (2019)). Among our variants, even though
they perform similarly in our experiments, Hessian based scaling (HNS) outperforms Jacobian based
scaling (NJS) in majority of the cases and this difference is signiﬁcant for one step FGSM attacks. This
indicates that nonlinearity of the network indeed has some relationship to its adversarial robustness.

We use state of the art models trained for binary quantization (where all layers are quantized) for our
experimental evaluations. We provide adversarial attack parameters used for FGSM/PGD in Appendix
and for other attacks, we use default parameters used in Foolbox (Rauber et al. (2017)). For our
HNS variant, we sweep β from a range such that the hessian norm is maximized for each image, as
explained in Appendix. For our NJS variant, we set the value of ρ = 0.01. In fact, our attacks are not
very sensitive to ρ and we provide the ablation study in the Appendix. The PyTorch (Paszke et al.
(2017)) implementation of our algorithm will be released upon publication.

6.1 RESULTS

We ﬁrst compared the original PGD (L2/L∞) and FGSM attack with both versions (NJS and HNS)
of improved PGD++ and FGSM++ attack, on CIFAR-10/100 datasets with ResNet-18/50, VGG-16,
DenseNet-121 and MobileNet-V2 network architectures and the adversarial accuracies for different
BNN-WQ are reported in Table 2. Our PGD++ variants consistently outperform original PGD on
all networks on both datasets. Even being a gradient based attack, our proposed PGD++ (L2/L∞)
variants can in fact reach adversarial accuracy close to 0 on CIFAR-10 dataset, demystifying the fake
robustness binarized networks tend to exhibit due to poor signal propagation.

7

Under review as a conference paper at ICLR 2021

Adversarial Accuracy (%)

Network

REF
BC
GD-tanh
MD-tanh-S

FGSM

62.38
53.91
56.13
55.10

FGSM
β = 0.1

FGSM++

NJS

HNS

69.52
62.46
65.06
63.42

61.43
52.90
55.54
54.74

61.40
52.27
54.81
53.82

PGD

48.73
41.29
42.77
41.34

PGD
β = 0.1

61.27
54.24
56.78
54.22

Deep
Fool

51.01
42.65
44.78
43.46

BBA

PGD++

NJS

HNS

48.43
40.14
42.94
40.69

47.17
39.35
42.14
40.76

48.54
39.34
42.30
40.67

Table 4: Adversarial accuracy on the test set of CIFAR-10 with ResNet-18 for adversarially trained
REF and BNN-WQ using different quantization methods (BC, GD-tanh, MD-tanh-S). Our improved
attacks are compared against FGSM, L∞ bounded PGD, a heuristic choice of β = 0.1, DeepFool and
BBA. Albeit on adversarially trained networks, our methods outperform all the comparable methods.

Network

REF
BNN-WQ
BNN-WAQ
∗

REF
BNN-WQ

∗

Adversarial Accuracy (%)

FGSM

7.62
40.49
40.84

62.38
55.10

FGSM
(DLR)

19.48
19.72
41.78

66.39
59.14

FGSM++

NJS

HNS

5.55
3.46
19.46

61.43
54.74

5.35
2.51
19.09

61.40
53.82

PGD

0.00
26.98
8.57

48.73
41.34

PGD
(DLR)

0.00
0.00
4.57

PGD++

NJS

0.00
0.00
0.03

HNS

0.00
0.00
0.04

49.73
41.42

47.17
40.76

48.54
40.67

Table 5: Adversarial accuracy for REF, BNN-WQ, and BNN-WAQ trained on CIFAR-10 using ResNet-18.
Here ∗ denotes adversarially trained models. Both our NJS and HNS variants consistently outperform
L∞ bounded FGSM and PGD attack performed with Difference of Logits Ratio (DLR) loss instead of
cross entropy loss. Notice, FGSM and PGD attack with DLR loss (Croce & Hein (2020)) perform even
worse than their original form on adversarially trained models.

Similarly, for one step FGSM attack, our modiﬁed versions outperform original FGSM attacks by a
signiﬁcant margin consistently for both datasets on various network architectures. We would like
to point out such an improvement in the above two attacks is considerably interesting, knowing the
fact that FGSM, PGD with L∞ attacks only use the sign of the gradients so improved performance
indicates, our temperature scaling indeed makes some zero elements in the gradient nonzero. We
would like to point out here that one can use several random restarts to increase the success rate of
original form of FGSM/PGD attack further but to keep comparisons fair we use single random restart
for both original and modiﬁed attacks. Nevertheless, as it has been observed in Table 1 even with
20 random restarts PGD adversarial accuracies for BNNs cannot reach zero, whereas our proposed
PGD++ variants consistently achieve perfect success rate.

ImageNet. For other large scale datasets such as ImageNet, BNNs are hard to train with full
binarization of parameters and result in poor performance. Thus, most existing works (Yang et al.
(2019)) on BNNs keep the ﬁrst and the last layers ﬂoating point and introduce several layerwise scalars
to achieve good results on ImageNet. In such experimental setups, according to our experiments,
trained BNNs do not exhibit gradient masking issues or poor signal propagation and thus are easier
to attack using original FGSM/PGD attacks with complete success rate. In such experiments, our
modiﬁed versions perform equally well compared to the original forms of these attacks.

The adversarial accuracies of REF and BNN-WAQ trained on CIFAR-10 using ResNet-18/50, VGG-16
and DenseNet-121 for our variants against original counterparts are reported in Table 3. Overall, for
both REF and BNN-WAQ, our variants outperform the original counterparts consistently. Particularly
interesting, PGD++ variants improve the attack success rate on REF networks. This effectively
expands the applicability of our PGD++ variants and encourages to consider signal propagation of any
trained network to improve gradient based attacks. PGD++ with L∞ variants achieve near-perfect
success rate on all BNN-WAQs, again validating the hypotheses of fake robustness of BNNs.

To further demonstrate the efﬁcacy, we ﬁrst adversarially trained the BNN-WQs (quantized using
BC (Courbariaux et al. (2015)), GD-tanh/MD-tanh-S (Ajanthan et al. (2019b))) and ﬂoating point
networks in a similar manner as in Madry et al. (2017), using L∞ bounded PGD with T = 7

8

Under review as a conference paper at ICLR 2021

iterations, η = 2 and (cid:15) = 8. We report the adversarial accuracies of L∞ bounded attacks and our
variants on CIFAR-10 using ResNet-18 in Table 4. These results further strengthens the usefulness
of our proposed PGD++ variants. Moreover, with a heuristic choice of β = 0.1 to scale down the
logits before performing gradient based attacks performs even worse. Finally, even against stronger
attacks (DeepFool (Moosavi-Dezfooli et al. (2016)), BBA (Brendel et al. (2019))) under the same L∞
perturbation bound, our variants outperform consistently on these adversarially trained models. We
would like to point out that our variants have negligible computational overhead over the original
gradient based attacks, whereas stronger attacks are much slower in practice requiring 100-1000
iterations with an adversarial starting point (instead of random initial perturbation).

To illustrate the effectiveness of our proposed variants in improving signal propagation, we compare
against gradient based attacks performed using recently proposed Difference of Logits Ratio (DLR)
loss (Croce & Hein (2020)) that aims to avoid the issue of saturating error signals. We show these
experimental comparisons performed on ResNet-18 models trained on CIFAR-10 dataset in Table 5.
The attack parameters are same as used for the other experiments. It can be clearly observed that
in almost all cases our proposed variants are much better than original form of gradient based
attacks performed with DLR loss. The margin of difference is signiﬁcant in case of FGSM attack and
adversarial trained models. Infact, it is important to note that gradient based attacks with DLR loss
perform worse on adversarially trained models than the original form of gradient based attacks.

7 RELATED WORK

Adversarial examples are ﬁrst observed in Szegedy et al. (2014) and subsequently efﬁcient gradient
based attacks such as FGSM (Goodfellow et al. (2014)) and PGD (Madry et al. (2017)) are introduced.
There exist recent stronger attacks such as Moosavi-Dezfooli et al. (2016); Carlini & Wagner (2017);
Yao et al. (2019); Finlay et al. (2019); Brendel et al. (2019), however, compared to PGD, they are
much slower to be used for adversarial training in practice. For a comprehensive survey related to
adversarial attacks, we refer the reader to Chakraborty et al. (2018).

Some recent works focus on the adversarial robustness of BNNs (Bernhard et al. (2019); Sen et al.
(2020); Galloway et al. (2018); Khalil et al. (2019); Lin et al. (2019)), however, a strong consensus
on the robustness properties of quantized networks is lacking. In particular, while Galloway et al.
(2018) claims parameter quantized networks are robust to gradient based attacks based on empirical
evidence, (Lin et al. (2019)) shows activation quantized networks are vulnerable to such attacks
and proposes a defense strategy assuming the parameters are ﬂoating-point. Differently, Khalil
et al. (2019) proposes a combinatorial attack hinting that activation quantized networks would have
obfuscated gradients issue. Sen et al. (2020) shows ensemble of mixed precision networks to be more
robust than original ﬂoating point networks; however Tramer et al. (2020) later shows the presented
defense method can be attacked with minor modiﬁcation in the loss function. In short, although it
has been hinted that there might be some sort of gradient masking in BNNs (especially in activation
quantized networks), a thorough understanding is lacking on whether BNNs are robust, if not what
is the reason for the inferior performance of most commonly used gradient based attacks on binary
networks. We answer this question in this paper and introduce improved gradient based attacks.

8 CONCLUSION

In this work, we have shown that both BNN-WQ and BNN-WAQ tend to show a fake sense of robustness
on gradient based attacks due to poor signal propagation. To tackle this issue, we introduced our
two variants of PGD++ attack, namely NJS and HNS. Our proposed PGD++ variants not only possess
near-complete success rate on binarized networks but also outperform standard L∞ and L2 bounded
PGD attacks on ﬂoating point networks. We ﬁnally show improvement in attack success rate on
adversarially trained REF and BNN-WQ against stronger attacks (DeepFool and BBA). In future, we
intend to focus more on improving the robustness of the BNNs with provable robustness guarantees.

REFERENCES

Thalaiyasingam Ajanthan, Puneet K Dokania, Richard Hartley, and Philip HS Torr. Proximal

mean-ﬁeld for neural network quantization. ICCV, 2019a.

9

Under review as a conference paper at ICLR 2021

Thalaiyasingam Ajanthan, Kartik Gupta, Philip HS Torr, Richard Hartley, and Puneet K Dokania.
Mirror descent view for neural network quantization. arXiv preprint arXiv:1910.08237, 2019b.

Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. Square attack:
a query-efﬁcient black-box adversarial attack via random search. In European Conference on
Computer Vision, pp. 484–501. Springer, 2020.

Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of
security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420, 2018.

Yu Bai, Yu-Xiang Wang, and Edo Liberty. Proxquant: Quantized neural networks via proximal

operators. ICLR, 2019.

Rémi Bernhard, Pierre-Alain Moellic, and Jean-Max Dutertre. Impact of low-bitwidth quantization
on the adversarial robustness for embedded neural networks. In 2019 International Conference on
Cyberworlds (CW), pp. 308–315. IEEE, 2019.

Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan Ustyuzhaninov, and Matthias Bethge.
Accurate, reliable and fast robustness evaluation. In Advances in Neural Information Processing
Systems, pp. 12861–12871, 2019.

Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In IEEE

Symposium on Security and Privacy, 2017.

Anirban Chakraborty, Manaar Alam, Vishal Dey, Anupam Chattopadhyay, and Debdeep Mukhopad-

hyay. Adversarial attacks and defences: A survey. arXiv preprint arXiv:1810.00069, 2018.

Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. Binaryconnect: Training deep neural

networks with binary weights during propagations. NeurIPS, 2015.

Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble

of diverse parameter-free attacks. ICML, 2020.

Chris Finlay, Aram-Alexandre Pooladian, and Adam Oberman. The logbarrier adversarial attack:
making effective use of decision boundary information. In Proceedings of the IEEE International
Conference on Computer Vision, pp. 4862–4870, 2019.

Angus Galloway, Graham W. Taylor, and Medhat Moussa. Attacking binarized neural networks. In

International Conference on Learning Representations, 2018.

Xavier Glorot and Yoshua Bengio. Understanding the difﬁculty of training deep feedforward neural
networks. In Proceedings of the thirteenth international conference on artiﬁcial intelligence and
statistics, pp. 249–256, 2010.

Ian Goodfellow. Gradient masking causes clever to overestimate adversarial perturbation size. arXiv

preprint arXiv:1804.07870, 2018.

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial

examples. arXiv preprint arXiv:1412.6572, 2014.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural
networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70,
pp. 1321–1330. JMLR. org, 2017.

Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Quantized
neural networks: Training neural networks with low precision weights and activations. JMLR,
2017.

Elias B Khalil, Amrita Gupta, and Bistra Dilkina. Combinatorial attacks on binarized neural networks.

In International Conference on Learning Representations, 2019.

Ji Lin, Chuang Gan, and Song Han. Defensive quantization: When efﬁciency meets robustness. In

International Conference on Learning Representations, 2019.

10

Under review as a conference paper at ICLR 2021

Yao Lu, Stephen Gould, and Thalaiyasingam Ajanthan. Bidirectional self-normalizing neural

networks. arXiv preprint arXiv:2006.12169, 2020.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.
Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083,
2017.

Mark D McDonnell. Training wide residual networks for deployment using a single bit for each

weight. ICLR, 2018.

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: a simple and
accurate method to fool deep neural networks. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pp. 2574–2582, 2016.

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Jonathan Uesato, and Pascal Frossard. Ro-
bustness via curvature regularization, and vice versa. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pp. 9078–9086, 2019.

Maria-Irina Nicolae, Mathieu Sinn, Minh Ngoc Tran, Beat Buesser, Ambrish Rawat, Martin Wistuba,
Valentina Zantedeschi, Nathalie Baracaldo, Bryant Chen, Heiko Ludwig, et al. Adversarial
robustness toolbox v1. 0.0. arXiv preprint arXiv:1807.01069, 2018.

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito,
Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in
PyTorch. 2017.

Jeffrey Pennington, Samuel Schoenholz, and Surya Ganguli. Resurrecting the sigmoid in deep
learning through dynamical isometry: theory and practice. In Advances in neural information
processing systems, pp. 4785–4795, 2017.

Chongli Qin, James Martens, Sven Gowal, Dilip Krishnan, Krishnamurthy Dvijotham, Alhussein
Fawzi, Soham De, Robert Stanforth, and Pushmeet Kohli. Adversarial robustness through local
linearization. In Advances in Neural Information Processing Systems, pp. 13824–13833, 2019.

Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. Xnor-net: Imagenet

classiﬁcation using binary convolutional neural networks. ECCV, 2016.

Jonas Rauber, Wieland Brendel, and Matthias Bethge. Foolbox: A python toolbox to benchmark the

robustness of machine learning models. arXiv preprint arXiv:1707.04131, 2017.

Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics

of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.

Sanchari Sen, Balaraman Ravindran, and Anand Raghunathan. Empir: Ensembles of mixed precision
deep networks for increased robustness against adversarial attacks. In International Conference on
Learning Representations, 2020.

Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow,
and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning
Representations, 2014.

Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick Mc-
Daniel. Ensemble adversarial training: Attacks and defenses. arXiv preprint arXiv:1705.07204,
2017.

Florian Tramer, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to

adversarial example defenses. arXiv preprint arXiv:2002.08347, 2020.

Tsui-Wei Weng, Huan Zhang, Pin-Yu Chen, Jinfeng Yi, Dong Su, Yupeng Gao, Cho-Jui Hsieh, and
Luca Daniel. Evaluating the robustness of neural networks: An extreme value theory approach. In
International Conference on Learning Representations, 2018.

Jiwei Yang, Xu Shen, Jun Xing, Xinmei Tian, Houqiang Li, Bing Deng, Jianqiang Huang, and
Xian-sheng Hua. Quantization networks. In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pp. 7308–7316, 2019.

11

Under review as a conference paper at ICLR 2021

Zhewei Yao, Amir Gholami, Qi Lei, Kurt Keutzer, and Michael W Mahoney. Hessian-based analysis
of large batch training and robustness to adversaries. In Advances in Neural Information Processing
Systems, pp. 4949–4959, 2018.

Zhewei Yao, Amir Gholami, Peng Xu, Kurt Keutzer, and Michael W Mahoney. Trust region based
adversarial attack on neural networks. In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pp. 11350–11359, 2019.

Appendices

Here, we ﬁrst provide the pseudocodes, proof of the proposition and the derivation of Hessian. Later
we give additional experiments, analysis and the details of our experimental setting.

A PSEUDOCODE

We provide pseudocode for PGD++ with NJS in Algorithm 1 and PGD++ with HNS in Algorithm 2.

Algorithm 1 PGD++ with NJS with L∞, T iterations, radius (cid:15), step size η, network fw∗ , input
x0, label k, one-hot y ∈ {0, 1}d, gradient threshold ρ.
Require: T, (cid:15), η, ρ, x0, y, k
Ensure: (cid:107)xT +1 − x0(cid:107)∞ ≤ (cid:15)
1: β1 = (M d)/(cid:0) (cid:80)M
(cid:80)d
2: x1 = P (cid:15)
3: for t ← 1, . . . T do
β2 = 1.0
4:
p(cid:48) = softmax(β1(fw∗ (xt)))
5:
if 1 − p(cid:48)
k ≤ ρ then
6:
7:

i=1
∞(x0 + Uniform(−1, 1))

(cid:46) β1 computed using Network Jacobian.
(cid:46) Random Initialization with Projection

(cid:46) ρ = 0.01
(cid:46) γ computed using Proposition 2

j=1 µj(Ji)(cid:1)

β2 = − log(ρ/(d − 1)(1 − ρ))/γ
(cid:96) = −yT log(softmax(β2β1(fw∗ (xt))))
xt+1 = P (cid:15)

∞(xt + η sign(∇x(cid:96)(xt)))

8:
9:

(cid:46) Update Step with Projection

Algorithm 2 PGD++ with HNS with L∞, T iterations, radius (cid:15), step size η, network fw∗ , input
x0, label k, one-hot y ∈ {0, 1}d, gradient threshold ρ.
Require: T, (cid:15), η, x0, y, k
Ensure: (cid:107)xT +1 − x0(cid:107)∞ ≤ (cid:15)
1: x1 = P (cid:15)
2: β∗ = argmaxβ>0
3: for t ← 1, . . . T do
4:
5:

(cid:96) = −yT log(softmax(β∗(fw∗ (xt))))
xt+1 = P (cid:15)

∞(x0 + Uniform(−1, 1))
(cid:13)∂2(cid:96)(β)/∂(x0)2(cid:13)
(cid:13)
(cid:13)F

(cid:46) Random Initialization with Projection
(cid:46) Grid Search

(cid:46) Update Step with Projection

∞(xt + η sign(∇x(cid:96)(xt)))

B DERIVATIONS

B.1 DERIVING β GIVEN A LOWERBOUND ON 1 − pk(β)

Proposition 2. Let aK ∈ IRd with d > 1 and aK
d and aK
For a given 0 < ρ < (d − 1)/d, there exists a β > 0 such that 1 − softmax(βaK
β < − log(ρ/(d − 1)(1 − ρ))/γ.

2 ≥ . . . ≥ aK

1 ≥ aK

1 − aK
d = γ.
1 ) > ρ, then

12

Under review as a conference paper at ICLR 2021

Proof. Assuming aK
1 − softmax(βaK

1 ) > ρ.

1 − aK

d = γ, we derive a condition on β such that

1 − softmax(βaK
softmax(βaK

1 ) > ρ ,
1 ) < 1 − ρ ,

exp(βaK

1 )/

d
(cid:88)

λ=1

exp(βaK

λ ) < 1 − ρ ,

1/(cid:0)1 +

d
(cid:88)

λ=2

exp(β(aK

λ − aK

1 ))(cid:1) < 1 − ρ .

Since, aK

1 − aK

λ ≤ γ for all λ > 1,

1/(cid:0)1 +

d
(cid:88)

λ=2

exp(β(aK

λ − aK

1 ))(cid:1) ≤ 1/(cid:0)1 +

d
(cid:88)

λ=2

exp(−βγ)(cid:1) .

Therefore, to ensure 1/(cid:0)1 + (cid:80)d

λ=2 exp(β(aK

λ − aK

1 ))(cid:1) < 1 − ρ, we consider,

1/(cid:0)1 +

d
(cid:88)

λ=2

exp(−βγ)(cid:1) < 1 − ρ ,

1 − aK
aK

λ ≤ γ for all λ > 1 ,

1/(cid:0)1 + (d − 1) exp(−βγ)(cid:1) < 1 − ρ ,

exp(−βγ) > ρ/(d − 1)(1 − ρ) ,

−βγ > log(ρ/(d − 1)(1 − ρ)) ,

exp is monotone ,

β < − log(ρ/(d − 1)(1 − ρ))/γ .

(9)

(10)

(11)

Therefore for any β < − log(ρ/(d − 1)(1 − ρ))/γ, the above inequality
1 − softmax(βaK

1 ) > ρ is satisﬁed.

B.2 DERIVATION OF HESSIAN

We now derive the Hessian of the input mentioned in Eq. (8) of the paper. The input gradients can be
written as:

(12)

(13)

∂(cid:96)(β)
∂x0 =
Now by product rule of differentiation, input hessian can be written as:

∂p(β)
∂¯aK(β)

βJ = ψ(β)βJ .

∂(cid:96)(β)
∂p(β)

∂2(cid:96)(β)
∂(x0)2 = β

(cid:34)

(cid:34)

ψ(β)

∂J
∂x0 +

(cid:18) ∂ψ(β)
∂x0

(cid:19)T

(cid:35)

J

,

= β

ψ(β)

∂J
∂x0 +

(cid:18) ∂p(β)
∂x0

(cid:19)T

(cid:35)

J

, ψ(β) = −(y − p(β))T ,

(cid:34)

= β

ψ(β)

∂J
∂x0 + β

(cid:18) ∂p(β)
∂¯aK J

(cid:19)T

(cid:35)

J

.

C ADDITIONAL EXPERIMENTS

In this section we ﬁrst provide more experimental details and then some ablation studies.

C.1 EXPERIMENTAL DETAILS

13

Under review as a conference paper at ICLR 2021

ResNet-18

VGG-16

Method

REF
BNN-WQ
BNN-WAQ

APGD

0.00
0.00
6.32

Square
Attack

PGD++

NJS

HNS

0.55
0.41
21.45

0.00
0.00
0.03

0.00
0.00
0.04

APGD

0.79
8.23
0.38

Square
Attack

PGD++

NJS

HNS

2.25
1.98
16.67

0.00
0.00
0.01

0.00
0.00
0.02

Table 7: Adversarial accuracy for REF, BNN-WQ and BNN-WAQ trained on CIFAR-10 using ResNet-18.
Both our NJS and HNS variants consistently outperform Auto-PGD (APGD) (Croce & Hein (2020))
performed using Difference of Logits Ratio (DLR) loss and a gradient free attack namely, Square
Attack (Andriushchenko et al. (2020)) under L∞ bound (8/255).

(cid:15)

η

T

Attack

Dataset

8
2
15

1
20
20

CIFAR-10

8
8
120

FGSM
PGD (L∞)
PGD (L2)

We ﬁrst mention the hyperparameters used to perform
FGSM and PGD attack for all the experiments in the
paper in Table 6. To make a fair comparison, we keep
the attack parameters same for our proposed variants
of FGSM++ and PGD++ attacks. For PGD++ with HNS
variant, we maximize Frobenius norm of Hessian with
respect to the input as speciﬁed in Eq. (8) of the pa-
per by grid search for the optimum β. We would like
to point out that since only ψ(β) and p(β) terms are
dependent on β, we do not need to do forward and
backward pass of the network multiple times during the grid search. This signiﬁcantly reduces the
computational overhead during the grid search. We can simply use the same network outputs aK
and network jacobian J (as computed without using β) for the grid search, while computing the
other terms at each iteration of grid search. We apply grid search to ﬁnd the optimum beta between
100 equally spaced intervals of β starting from β1 to β2. Here, β1 and β2 are computed based on
Proposition 1 in the paper where ρ = 1e − 72 and ρ = 1 − (1/d) − (1e − 2) respectively, where d is
number of classes and γ = aK
2 so that 1 − softmax(βaK
1 ) < ρ. Also, note that we estimate
the optimum β for each test sample only at the start of the ﬁrst iteration of an iterative attack and then
use the same β for the next iterations.

Table 6: Attack parameters ((cid:15) & η in pixels).

FGSM
PGD (L∞)
PGD (L2)

1 − aK

CIFAR-100

1
10
10

4
4
60

4
1
15

Computational Overhead of NJS and HNS. Our Jacobian calculation takes just a single backward
pass through the network and thus adds a negligible overhead. Our NJS approach for scaling estimates
β as inverse of mean JSV using 100 random test samples, which is similar to 100 backward passes.
For HNS, in Eq. (8) Jacobian J can be computed in single backward pass. Moreover, for piecewise
linear networks (eg, relu activations), ∂J/∂x0 = 0 almost everywhere (Yao et al. (2018)). Thus
PGD++ with NJS and HNS is almost as efﬁcient as PGD.

C.2 COMPARISONS AGAINST AUTO-PGD ATTACK AND GRADIENT FREE ATTACK

We also compared our proposed PGD++ variants against recently proposed Auto-PGD (APGD) with
Difference of Logits Ratio (DLR) loss (Croce & Hein (2020)) and gradient free Square Attack (An-
driushchenko et al. (2020)) on different networks trained using ResNet-18 and VGG-16 on CIFAR-10
dataset and the results are reported in Table 7. The attack parameters for this experiment are the same
as reported in the paper. It can be clearly seen that our proposed variants perform much better than
both APGD with DLR loss and Square Attack, consistently achieving 0% adversarial accuracy. Infact,
much computationally expensive Square attack is unable to achieve 0% adversarial accuracy in any
of the cases under the enforced L∞ bound.

C.3 OTHER EXPERIMENTS

We provide adversarial accuracy comparisons for different attack methods on CIFAR-100 using ResNet-
18, VGG-16, ResNet-50 and DenseNet-121 in Table 8. Again similar to the results in the paper, our
proposed PGD++ and FGSM++ outperform original form of PGD and FGSM consistently in all the
experiments on ﬂoating point networks. We also provide adversarial accuracy comparison of our
proposed variants against stronger attacks namely DeepFool (Moosavi-Dezfooli et al. (2016)) and

14

Under review as a conference paper at ICLR 2021

Network

ResNet-18
VGG-16
ResNet-50
DenseNet-121

FGSM

9.06
16.28
12.95
11.41

FGSM++

NJS

HNS

9.23
17.24
12.95
11.41

2.70
9.19
11.94
10.74

Adversarial Accuracy (%)

PGD (L∞)

PGD++ (L∞)
HNS
NJS

PGD (L2)

0.14
1.53
0.12
0.00

0.14
0.95
0.00
0.00

0.00
0.25
0.00
0.00

5.38
4.87
31.01
6.10

PGD++ (L2)
HNS
NJS

0.17
1.50
4.43
3.09

0.15
1.38
4.14
2.76

Table 8: Adversarial accuracy on the test set of CIFAR-100 for REF (ﬂoating point networks). Both our
NJS and HNS variants consistently outperform original FGSM and PGD (L∞/L2 bounded) attacks.

Network

PGD

ResNet-18
VGG-16

8.57
78.01

Deep
Fool

18.92
12.12

BBA

PGD++

NJS

HNS

0.81
0.10

0.03
0.01

0.04
0.02

Table 9: Adversarial accuracy on the test set of CIFAR-10 for BNN-WAQ. Here, we compare our
proposed variants against much stronger attacks namely DeepFool (Moosavi-Dezfooli et al. (2016))
and BBA (Brendel et al. (2019)). Both our variants outperform stronger attacks. Note, DeepFool and
BBA are much slower in practise requiring 100-1000 iterations. BBA speciﬁcally requires even an
adversarial start point that needs to be computed using another adversarial attack.

BBA (Brendel et al. (2019)) on BNN-WAQ trained on CIFAR-10 dataset in Table 9. In this experiment,
our proposed variants again outperform even the stronger attacks which take 100-1000 iterations with
adversarial start point (instead of random initial perturbation). It should be noted that although BBA
performs much better than DeepFool and PGD, it still has inferior success rate than ours considering
the fact that it takes multiple hours to run BBA whereas our proposed variants are almost as efﬁcient
as PGD attack.

Step Size Tuning for PGD attack. We would like to point out that step size η and temperature scale
β have different effects in the attacks performed. Notice, PGD and FGSM attack under L∞ bound only
use the sign of input gradients in each gradient ascent step. Thus, if the input gradients are completely
saturated (which is the case for BNNs), original forms of PGD or FGSM will not work irrespective of
the step size used. To illustrate this, we performed extensive step size tuning for original form of PGD
attack on different ResNet-18 models trained on CIFAR-10 dataset and the adversarial accuracies are
reported in Fig. 4. It can be observed clearly that although tuning the step size lowers adversarial
accuracy a bit in some cases but still cannot reach zero for BNNs unlike our proposed variants.

Adversarial training using PGD++. We also investigate
the potential application of PGD++ for adversarial training
to improve the robustness of neural networks. PGD++
attack is most effective when applied to a network with
poor signal propagation. However, adversarial training is
performed from random initialization (Glorot & Bengio
(2010)) exhibiting good signal propagation. Thus, PGD
and PGD++ perform similarly for adversarial training. We
infer these conclusions from our experiments on adversarial
training using PGD++.

CLEVER Scores. Recently CLEVER Scores (Weng et al.
(2018)) have been proposed as an empirical estimate to
measure robustness lower bounds for deep networks. It
has been later shown that gradient masking issues cause
CLEVER to overestimate the robustness bounds (Goodfellow (2018)). Here we try to improve the
CLEVER scores using different ways of choosing β in temperature scaling. For this experiment,
we use CLEVER implementation of Adversarial Training Toolbox2 (Nicolae et al. (2018)). We set

Figure 4: Adversarial accuracy using
PGD attack under L∞ bound (8/255)
with varying step size (η) on ResNet-18
trained on CIFAR-10. Notice, PGD at-
tack is unable to reach zero adversarial
accuacy for BNNs with any step size.

2https://github.com/Trusted-AI/adversarial-robustness-toolbox

15

0510152025303540Attack Step Size0102030405060Adversarial accuracy (in %)Varying PGD Attack Step SizeREFBNN­WQBNN­WAQUnder review as a conference paper at ICLR 2021

Original Heuristic

NJS

HNS

BNN-WQ
BNN-WAQ

0.8585
0.7239

0.8845
3.1578

0.4139
0.3120

0.3450
0.2774

Table 10: CLEVER Scores (Weng et al. (2018)) for BNN-WQ and BNN-WAQ trained on CIFAR-10 using
ResNet-18. We compare CLEVER Scores returned for L1 norm perturbation using different ways of
temperature scaling applied. Here, Original refers to original network without temperature scaling
and Heuristic denotes temperature scale with small β = 0.01.

Methods

REF
BNN-WQ
BNN-WAQ

1e − 05

1e − 04

1e − 03

1e − 02

1e − 01

2e − 01

PGD++ (NJS) - Varying ρ

0.00
0.00
0.15

0.00
0.00
0.08

0.00
0.00
0.04

0.00
0.00
0.03

0.00
0.00
0.04

0.00
0.00
0.02

Table 11: Adversarial accuracy on the test set for binary neural networks using L∞ bounded PGD++
attack using NJS with varying ρ. For different values of ρ, our approach is quite stable.

number of batches to 50, batch size to 10, radius to 5, and chose L1 norm as hyperparameters (based
on the Weng et al. (2018)). We compare our variants namely NJS and HNS against heuristic choice
of small β = 0.01 and original CLEVER Scores for BNN-WQ and BNN-WAQ (trained on CIFAR-10
using ResNet-18) in Table 10. It can be clearly seen that our proposed variants improve the robustness
bounds computed using CLEVER whereas a heuristic choice of β = 0.01 performs even worse.

C.4 STABILITY OF PGD++ WITH NJS WITH VARIATIONS IN ρ

We perform ablation studies with varying ρ for PGD++ with NJS in Table 11 for CIFAR-10 dataset
using ResNet-18 architecture. It clearly illustrates that our NJS variant is quite robust to the choice of
ρ as we are able to achieve near perfect success rate with PGD++ with different values of ρ. As long
as value of ρ is large enough to avoid one-hot encoding on softmax outputs (in turn avoid (cid:107)ψ(β)(cid:107) to
be zero) of correctly classiﬁed sample, our approach with NJS variant is quite stable.

C.5 SIGNAL PROPAGATION AND INPUT GRADIENT ANALYSIS USING NJS AND HNS

We ﬁrst provide an example illustration in Fig. 5 to better understand how the input gradient norm
i.e., (cid:107)∂(cid:96)(β)/∂x0(cid:107)2, and norm of sign of input gradient, i.e., (cid:107)sign(∂(cid:96)(β)/∂x0)(cid:107)2 is inﬂuenced by
β. It clearly shows that both the plots have a concave behavior where an optimal β can maximize
the input gradient. Also, it can be quite evidently seen in Fig. 5 (b) that within an optimal range of
β, gradient vanishing issue can be avoided. If β → 0 or β → ∞, it changes all the values in input
gradient matrix to zero and inturn (cid:107)sign(∂(cid:96)(β)/∂x0)(cid:107)2 = 0.

We also provide the signal propagation properties as well as analysis on input gradient norm before
and after using the β estimated based on NJS and HNS in Table 12. For binarized networks as well
ﬂoating point networks tested on CIFAR-10 dataset using ResNet-18 architecture, our HNS and NJS
variants result in larger values for (cid:107)ψ(cid:107)2, (cid:107)∂(cid:96)(β)/∂x0(cid:107)2 and (cid:107)sign(∂(cid:96)(β)/∂x0)(cid:107)2. This reﬂects the
efﬁcacy of our method in overcoming the gradient vanishing issue. It can be also noted that our
variants also improves the signal propagation of the networks by bringing the mean JSV values closer
to 1.
C.6 ABLATION FOR ρ VS. PGD++ ACCURACY

In this subsection, we provide the analysis on the effect of bounding the gradients of the network
output of ground truth class k, i.e. ∂(cid:96)(β)/∂¯aK
k . Here, we compute β using Proposition 1 for all
correctly classiﬁed images such that 1 − softmax(βaK
k ) > ρ with different values of ρ and report
the PGD++ adversarial accuracy in Table 13. It can be observed that there is an optimum value of
ρ at which PGD++ success rate is maximized, especially on the adversarially trained models. This
can also be seen in connection with the non-linearity of the network where at an optimum value of

16

Under review as a conference paper at ICLR 2021

(a)

(b)

Figure 5: Plots to show how variation in β affects (a) norm of input gradient, i.e., (cid:107)∂(cid:96)(β)/∂x0(cid:107)2,
(b) norm of sign of input gradient, i.e., (cid:107)sign(∂(cid:96)(β)/∂x0)(cid:107)2 on a random correctly classiﬁed image.
Notice that, both input gradient and signed input gradient norm behave similarly, showing a concave
behaviour. This plot is computed for BNN-WQ network on CIFAR-10, ResNet-18. (b) clearly illustrates
how optimum β can avoid vanishing gradient issue since (cid:107)sign(∂(cid:96)(β)/∂x0)(cid:107)2 will only be zero if
input gradient matrix has only zeros.

Methods

REF

Adv. Train

BNN-WQ

BNN-WAQ

JSV (Mean)

JSV (Std.)

(cid:107)ψ(cid:107)2
(cid:107)ψ(cid:107)2
(cid:107)ψ(cid:107)2

(cid:107)∂(cid:96)/∂x0(cid:107)2
(cid:107)∂(cid:96)/∂x0(cid:107)2
(cid:107)∂(cid:96)/∂x0(cid:107)2

(cid:107)sign(cid:0) ∂(cid:96)
(cid:107)sign(cid:0) ∂(cid:96)
(cid:107)sign(cid:0) ∂(cid:96)
∂x0
∂x0
∂x0

(cid:1)(cid:107)2
(cid:1)(cid:107)2
(cid:1)(cid:107)2

Orig.
NJS
HNS

Orig.
NJS
HNS

Orig.
NJS
HNS

Orig.
NJS
HNS

Orig.
NJS
HNS

8.09e+00
9.51e−01
2.38e+00

6.27e+00
7.58e−01
4.41e+00

9.08e−03
4.66e−01
1.48e−01

2.42e−01
9.52e−01
7.49e−01

5.55e+01
5.55e+01
5.55e+01

5.15e−01
5.70e−01
6.11e+00

4.10e−01
6.34e−01
5.34e+02

2.33e−01
2.35e−01
2.57e−01

8.52e−02
1.10e−01
8.18e−01

5.54e+01
5.54e+01
5.54e+01

3.53e+01
9.95e−01
1.19e+01

3.53e+01
9.71e−01
2.13e+02

6.20e−03
5.37e−01
2.07e−01

2.27e−01
8.91e−01
3.70e−01

4.39e+01
5.55e+01
5.55e+01

1.11e+00
2.24e−01
4.65e+00

1.97e+00
6.73e−01
1.24e+02

9.46e−03
1.20e−01
2.44e−01

6.33e−02
1.24e−01
2.70e−01

5.55e+01
5.55e+01
5.55e+01

Table 12: Mean and standard deviation of Jacobian Singular Values (JSV), mean (cid:107)ψ(cid:107)2, mean
(cid:107)∂(cid:96)/∂x0(cid:107)2 and mean (cid:107)sign(∂(cid:96)/∂x0)(cid:107)2 for different methods on CIFAR-10 with ResNet-18 computed
with 500 correctly classiﬁed samples. Note here for NJS and HNS, JSV is computed for scaled jacobian
i.e. βJ. Also note that, values of (cid:107)ψ(cid:107)2, (cid:107)∂(cid:96)(β)/∂x0(cid:107)2 and (cid:107)sign(∂(cid:96)(β)/∂x0)(cid:107)2 are larger for our
NJS and HNS variant (for most of the networks) as compared with network with no β, which clearly
indicates better gradients for performing gradient based attacks.

β, even for robust (locally linear) (Moosavi-Dezfooli et al. (2019); Qin et al. (2019)) networks such
as adversarially trained models, non-linearity can be maximized and better success rate for gradient
based attacks can be achieved. Our HNS variant essentially tries to achieve the same objective while
trying to estimate β for each example.

17

0.000.050.100.150.200.250.300.00.20.40.60.81.0||/x0||01234501020304050||sign(/x0)||Under review as a conference paper at ICLR 2021

Methods

REF
BNN-WQ
REF∗
BNN-WQ∗

PGD++ with Varying ρ

1e − 15

1e − 09

1e − 05

1e − 01

2e − 01

5e − 01

0.00
9.61

48.18
40.66

0.00
0.04

47.66
40.01

0.00
0.00

48.00
40.04

0.00
0.00

53.09
45.09

0.00
0.00

54.58
46.57

0.00
0.00

57.57
49.72

Table 13: Adversarial accuracy on the test set for adversarially trained networks and binary neural
networks using L∞ bounded PGD++ attack with varying ρ as lower bound on the gradient of network
output for ground truth class k. Here * denotes the adversarially trained models obtained where
adversarial samples are generated using L∞ bounded PGD attack with with T = 7 iterations, η = 2
and (cid:15) = 8. Note, here PGD++ attack refers to PGD attack where ∂(cid:96)(β)/∂¯aK
k is bounded by ρ for
each sample, where k is ground truth class.

18

