Under review as a conference paper at ICLR 2021

SPARSE LINEAR NETWORKS WITH A FIXED BUTTER-
FLY STRUCTURE: THEORY AND PRACTICE

Anonymous authors
Paper under double-blind review

ABSTRACT

A butterﬂy network consists of logarithmically many layers, each with a lin-
ear number of non-zero weights (pre-speciﬁed). The fast Johnson-Lindenstrauss
transform (FJLT) can be represented as a butterﬂy network followed by a projec-
tion onto a random subset of the coordinates. Moreover, a random matrix based
on FJLT with high probability approximates the action of any matrix on a vec-
tor. Motivated by these facts, we propose to replace a dense linear layer in any
neural network by an architecture based on the butterﬂy network. The proposed
architecture signiﬁcantly improves upon the quadratic number of weights required
in a standard dense layer to nearly linear with little compromise in expressibility
of the resulting operator. In a collection of wide variety of experiments, includ-
ing supervised prediction on both the NLP and vision data, we show that this
not only produces results that match and often outperform existing well-known
architectures, but it also offers faster training and prediction in deployment. To
understand the optimization problems posed by neural networks with a butterﬂy
network, we study the optimization landscape of the encoder-decoder network,
where the encoder is replaced by a butterﬂy network followed by a dense linear
layer in smaller dimension. Theoretical result presented in the paper explain why
the training speed and outcome are not compromised by our proposed approach.
Empirically we demonstrate that the network performs as well as the encoder-
decoder network.

1

INTRODUCTION

A butterﬂy network (see Figure 6 in Appendix A) is a layered graph connecting a layer of n inputs to
a layer of n outputs with O(log n) layers, where each layer contains 2n edges. The edges connecting
adjacent layers are organized in disjoint gadgets, each gadget connecting a pair of nodes in one layer
with a corresponding pair in the next layer by a complete graph. The distance between pairs doubles
from layer to layer. This network structure represents the execution graph of the Fast Fourier Trans-
form (FFT) (Cooley and Tukey, 1965), Walsh-Hadamard transform, and many important transforms
in signal processing that are known to have fast algorithms to compute matrix-vector products.

Ailon and Chazelle (2009) showed how to use the Fourier (or Hadamard) transform to perform
fast Euclidean dimensionality reduction with Johnson and Lindenstrauss (1984) guarantees. The
resulting transformation, called Fast Johnson Lindenstrauss Transform (FJLT), was improved in
subsequent works (Ailon and Liberty, 2009; Krahmer and Ward, 2011). The common theme in
this line of work is to deﬁne a fast randomized linear transformation that is composed of a random
diagonal matrix, followed by a dense orthogonal transformation which can be represented via a
butterﬂy network, followed by a random projection onto a subset of the coordinates (this research is
still active, see e.g. Jain et al. (2020)). In particular, an FJLT matrix can be represented (explicitly)
by a butterﬂy network followed by projection onto a random subset of coordinates (a truncation
operator). We refer to such a representation as a truncated butterﬂy network (see Section 4).
Simple Johnson-Lindenstrauss like arguments show that with high probability for any W ∈ Rn2×n1
and any x ∈ Rn1 , W x is close to (J T
1 J1)x where J1 ∈ Rk1×n1 and J2 ∈ Rk2×n2
are both FJLT, and k1 = log n1, k2 = log n2 (see Section 4.2 for details). Motivated by this, we
propose to replace a dense (fully-connected) linear layer of size n2 × n1 in any neural network
1 W (cid:48)J2, where J1, J2 can be represented by a truncated butterﬂy
by the following architecture: J T

2 J2)W (J T

1

Under review as a conference paper at ICLR 2021

network and W (cid:48) is a k2 × k1 dense linear layer. The clear advantages of such a strategy are: (1)
almost all choices of the weights from a speciﬁc distribution, namely the one mimicking FJLT,
preserve accuracy while reducing the number of parameters, and (2) the number of weights is
nearly linear in the layer width of W (the original matrix). Our empirical results demonstrate that
this offers faster training and prediction in deployment while producing results that match and
often outperform existing known architectures. Compressing neural networks by replacing linear
layers with structured linear transforms that are expressed by fewer parameters have been studied
extensively in the recent past. We compare our approach with these related works in Section 3.

Since the butterﬂy structure adds logarithmic depth to the architecture, it might pose optimization
related issues. Moreover, the sparse structure of the matrices connecting the layers in a butterﬂy
network deﬁes the general theoretical analysis of convergence of deep linear networks. We take
a small step towards understanding these issues by studying the optimization landscape of a
encoder-decoder network (two layer linear neural network), where the encoder layer is replaced by
a truncated butterﬂy network followed by a dense linear layer in fewer parameters. This replacement
is motivated by the result of Sarl´os (2006), related to fast randomized low-rank approximation
of matrices using FJLT (see Section 4.2 for details). We consider this replacement instead of
the architecture consisting of two butterﬂy networks and a dense linear layer as proposed earlier,
because it is easier to analyze theoretically. We also empirically demonstrate that our new network
with fewer parameters performs as well as an encoder-decoder network.

The encoder-decoder network computes the best low-rank approximation of the input matrix. It
is well-known that with high probability a close to optimal low-rank approximation of a matrix
is obtained by either pre-processing the matrix with an FJLT (Sarl´os, 2006) or a random sparse
matrix structured as given in Clarkson and Woodruff (2009) and then computing the best low-rank
approximation from the rows of the resulting matrix1. A recent work by Indyk et al. (2019) studies
this problem in the supervised setting, where they ﬁnd the best pre-processing matrix structured as
given in Clarkson and Woodruff (2009) from a sample of matrices (instead of using a random sparse
matrix). Since an FJLT can be represented by a truncated butterﬂy network, we emulate the setting
of Indyk et al. (2019) but learn the pre-processing matrix structured as a truncated butterﬂy network.

2 OUR CONTRIBUTION AND POTENTIAL IMPACT

We provide an empirical report, together with a theoretical analysis to justify our main idea of
using sparse linear layers with a ﬁxed butterﬂy network in deep learning. Our ﬁndings indicate that
this approach, which is well rooted in the theory of matrix approximation and optimization, can
offer signiﬁcant speedup and energy saving in deep learning applications. Additionally, we believe
that this work would encourage more experiments and theoretical analysis to better understand the
optimization and generalization of our proposed architecture (see Future Work section).

On the empirical side – The outcomes of the following experiments are reported:

(1) In Section 6.1, we replace a dense linear layer in the standard state-of-the-art networks, for
both image and language data, with an architecture that constitutes the composition of (a) truncated
butterﬂy network, (b) dense linear layer in smaller dimension, and (c) transposed truncated butterﬂy
network (see Section 4.2). The structure parameters are chosen so as to keep the number of weights
near linear (instead of quadratic).

(2) In Sections 6.2 and 6.3, we train a linear encoder-decoder network in which the encoder is
replaced by a truncated butterﬂy network followed by a dense linear layer in smaller dimension.
These experiments support our theoretical result. The network structure parameters are chosen so
as to keep the number of weights in the (replaced) encoder near linear in the input dimension. Our
results (also theoretically) demonstrate that this has little to no effect on the performance compared
to the standard encoder-decoder network.

(3) In Section 7, we learn the best pre-processing matrix structured as a truncated butterﬂy network
to perform low-rank matrix approximation from a given sample of matrices. We compare our results

1The pre-processing matrix is multiplied from the left.

2

Under review as a conference paper at ICLR 2021

to that of Indyk et al. (2019), which learn the pre-processing matrix structured as given in Clarkson
and Woodruff (2009).

On the theoretical side – The optimization landscape of linear neural networks with dense matrices
have been studied by Baldi and Hornik (1989), and Kawaguchi (2016). The theoretical part of
this work studies the optimization landscape of the linear encoder-decoder network in which the
encoder is replaced by a truncated butterﬂy network followed by a dense linear layer in smaller
dimension. We call such a network as the encoder-decoder butterﬂy network. We give an overview
of our main result, Theorem 1, here. Let X ∈ Rn×d and Y ∈ Rm×d be the data and output
matrices respectively. Then the encoder-decoder butterﬂy network is given as Y = DEBX, where
D ∈ Rm×k and E ∈ Rk×(cid:96) are dense layers, B is an (cid:96) × n truncated butterﬂy network (product
of log n sparse matrices) and k ≤ (cid:96) ≤ m ≤ n (see Section 5). The objective is to learn D, E
and B that minimizes ||Y − Y ||2
F. Theorem 1 shows how the loss at the critical points of such
a network depends on the eigenvalues of the matrix Σ = Y X T BT (BXX T BT )−1BXY T 2. In
comparison, the loss at the critical points of the encoder-decoder network (without the butterﬂy
network) depends on the eigenvalues of the matrix Σ(cid:48) = Y X T (XX T )−1XY T (Baldi and Hornik,
1989). In particular, the loss depends on how the learned matrix B changes the eigenvalues of Σ(cid:48). If
we learn only for an optimal D and E, keeping B ﬁxed (as done in the experiment in Section 6.3)
then it follows from Theorem 1 that every local minima is a global minima and that the loss at the
local/global minima depends on how B changes the top k eigenvalues of Σ(cid:48). This inference together
with a result by Sarl´os (2006) is used to give a worst-case guarantee in the special case when Y = X
(called auto-encoders that capture PCA; see the below Theorem 1).

3 RELATED WORK

Important transforms like discrete Fourier, discrete cosine, Hadamard and many more satisfy a prop-
erty called complementary low-rank property, recently deﬁned by Li et al. (2015). For an n × n ma-
trix satisfying this property related to approximation of speciﬁc sub-matrices by low-rank matrices,
Michielssen and Boag (1996) and O’Neil et al. (2010) developed the butterﬂy algorithm to compute
the product of such a matrix with a vector in O(n log n) time. The butterﬂy algorithm factorizes such
a matrix into O(log n) many matrices, each with O(n) sparsity. In general, the butterﬂy algorithm
has a pre-computation stage which requires O(n2) time (O’Neil et al., 2010; Seljebotn, 2012). With
the objective of reducing the pre-computation cost Li et al. (2015); Li and Yang (2017) compute
the butterﬂy factorization for an n × n matrix satisfying the complementary low-rank property in
O(n 3
2 ) time. This line of work does not learn butterﬂy representations for matrices or apply it in
neural networks, and is incomparable to our work.

A few works in the past have used deep learning models with structured matrices (as hidden layers).
Such structured matrices can be described using fewer parameters compared to a dense matrix, and
hence a representation can be learned by optimizing over a fewer number of parameters. Examples
of structured matrices used include low-rank matrices (Denil et al., 2013; Sainath et al., 2013),
circulant matrices (Cheng et al., 2015; Ding et al., 2017), low-distortion projections (Yang et al.,
2015), Toeplitz like matrices (Sindhwani et al., 2015; Lu et al., 2016; Ye et al., 2018), Fourier-related
transforms (Moczulski et al., 2016) and matrices with low-displacement rank (Thomas et al., 2018).
Recently Alizadeh et al. (2020) demonstrated the beneﬁts of replacing the pointwise convolutional
layer in CNN’s by a butterﬂy network. Other works by Mocanu et al. (2018); Lee et al. (2019); Wang
et al. (2020); Verdenius et al. (2020) consider a different approach to sparsify neural networks. The
works closest to ours are by Yang et al. (2015), Moczulski et al. (2016), and Dao et al. (2020) and
we make a comparison below.

Yang et al. (2015) and Moczulski et al. (2016) attempt to replace dense linear layers with a stack
of structured matrices, including a butterﬂy structure (the Hadamard or the Cosine transform), but
they do not place trainable weights on the edges of the butterﬂy structure as we do. Note that adding
these trainable weights does not compromise the run time beneﬁts in prediction, while adding to
the expressiveness of the network in our case. Dao et al. (2020) replace handcrafted structured sub-
networks in machine learning models by a kaleidoscope layer, which consists of compositions of
butterﬂy matrices. This is motivated by the fact that the kaleidoscope hierarchy captures a structured
matrix exactly and optimally in terms of multiplication operations required to perform the matrix

2At a critical point the gradient of the loss function with respect to the parameters in the network is zero.

3

Under review as a conference paper at ICLR 2021

vector product operation. Their work differs from us as we propose to replace any dense linear layer
in a neural network (instead of a structured sub-network) by the architecture proposed in Section 4.2.
Our approach is motivated by theoretical results which establish that this can be done with almost
no loss in representation.

Finally, Dao et al. (2019) show that butterﬂy representations of standard transformations like discrete
Fourier, discrete cosine, Hadamard mentioned above can be learnt efﬁciently. They additionally
show the following: a) for the benchmark task of compressing a single hidden layer model they
compare the network constituting of a composition of butterﬂy networks with the classiﬁcation
accuracy of a fully-connected linear layer and b) in ResNet a butterﬂy sub-network is added to get
an improved result. In comparison, our approach to replace a dense linear layer by the proposed
architecture in Section 4.2 is motivated by well-known theoretical results as mentioned previously,
and the results of the comprehensive list of experiments in Section 6.1 support our proposed method.

4 PROPOSED REPLACEMENT FOR A DENSE LINEAR LAYER

In Section 4.1, we deﬁne a truncated butterﬂy network, and in Section 4.2 we motivate and state
our proposed architecture based on truncated butterﬂy network to replace a dense linear layer in any
neural network. All logarithms are in base 2, and [n] denotes the set {1, . . . , n}.

4.1 TRUNCATED BUTTERFLY NETWORK

Deﬁnition 4.1 (Butterﬂy Network). Let n be an integral power of 2. Then an n×n butterﬂy network
B (see Figure 6) is a stack of of log n linear layers, where in each layer i ∈ {0, . . . , log n − 1}, a
bipartite clique connects between pairs of nodes j1, j2 ∈ [n], for which the binary representation of
j1 − 1 and j2 − 1 differs only in the i’th bit. In particular, the number of edges in each layer is 2n.

In what follows, a truncated butterﬂy network is a butterﬂy network in which the deepest layer is
truncated, namely, only a subset of (cid:96) neurons are kept and the remaining n − (cid:96) are discarded. The in-
teger (cid:96) is a tunable parameter, and the choice of neurons is always assumed to be sampled uniformly
at random and ﬁxed throughout training in what follows. The effective number of parameters (train-
able weights) in a truncated butterﬂy network is at most 2n log (cid:96) + 6n, for any (cid:96) and any choice of
neurons selected from the last layer.3 We include a proof of this simple upper bound in Appendix F
for lack of space (also, refer to Ailon and Liberty (2009) for a similar result related to computation
time of truncated FFT). The reason for studying a truncated butterﬂy network follows (for exam-
ple) from the works (Ailon and Chazelle, 2009; Ailon and Liberty, 2009; Krahmer and Ward, 2011).
These papers deﬁne randomized linear transformations with the Johnson-Lindenstrauss property and
an efﬁcient computational graph which essentially deﬁnes the truncated butterﬂy network. In what
follows, we will collectively denote these constructions by FJLT. 4

4.2 MATRIX APPROXIMATION USING BUTTERFLY NETWORKS

We begin with the following proposition, following known results on matrix approximation (proof
in Appendix B).
Proposition 1. Suppose J1 ∈ Rk1×n1 and J2 ∈ Rk2×n2 are matrices sampled from FJLT distribu-
tion, and let W ∈ Rn2×n1. Then for the random matrix W (cid:48) = (J T
1 J1), any unit vector
x ∈ Rn1 and any (cid:15) ∈ (0, 1), Pr [(cid:107)W (cid:48)x − W x(cid:107) ≤ (cid:15)(cid:107)W (cid:107)] ≥ 1 − e−Ω(min{k1,k2}(cid:15)2) .

2 J2)W (J T

From Proposition 1 it follows that W (cid:48) approximates the action of W with high probability on any
given input vector. Now observe that W (cid:48) is equal to J T
1 . Since J1 and
2
J2 are FJLT, they can be represented by a truncated butterﬂy network, and hence it is conceivable to
replace a dense linear layer connecting n1 neurons to n2 neurons (containing n1n2 variables) in any

˜W J1, where ˜W = J2W J T

3Note that if n is not a power of 2 then we work with the ﬁrst n columns of the (cid:96) × n(cid:48) truncated butterﬂy

network, where n(cid:48) is the closest number to n that is greater than n and is a power of 2.

4To be precise, the construction in Ailon and Chazelle (2009), Ailon and Liberty (2009), and Krahmer and
Ward (2011) also uses a random diagonal matrix, but the values of the diagonal entries can be ‘absorbed’ inside
the weights of the ﬁrst layer of the butterﬂy network.

4

Under review as a conference paper at ICLR 2021

neural network with a composition of three gadgets: a truncated butterﬂy network of size k1 × n1,
followed by a dense linear layer of size k2 × k1, followed by the transpose of a truncated butterﬂy
network of size k2 × n2. In Section 6.1, we replace dense linear layers in common deep learning
networks with our proposed architecture, where we set k1 = log n1 and k2 = log n2.

5 ENCODER-DECODER BUTTERFLY NETWORK

Let X ∈ Rn×d, and Y ∈ Rm×d be data and output matrices respectively, and k ≤ m ≤ n. Then
the encoder-decoder network for X is given as

Y = DEX
where E ∈ Rk×n, and D ∈ Rm×k are called the encoder and decoder matrices respectively. For
the special case when Y = X, it is called auto-encoders. The optimization problem is to learn
matrices D and E such that ||Y − Y ||2
F is minimized. The optimal solution is denoted as Y ∗, D∗
and E∗5. In the case of auto-encoders X ∗ = Xk, where Xk is the best rank k approximation of X.
In this section, we study the optimization landscape of the encoder-decoder butterﬂy network : an
encoder-decoder network, where the encoder is replaced by a truncated butterﬂy network followed
by a dense linear layer in smaller dimension. Such a replacement is motivated by the following
result from Sarl´os (2006), in which ∆k = ||Xk − X||2
F.
Proposition 2. Let X ∈ Rn×d. Then with probability at least 1/2, the best rank k approximation
of X from the rows of JX (denoted Jk(X)), where J is sampled from an (cid:96) × n FJLT distribution
and (cid:96) = (k log k + k/(cid:15)) satisﬁes ||Jk(X) − X||2

F ≤ (1 + (cid:15))∆k.

Proposition 2 suggests that in the case of auto-encoders we could replace the encoder with a trun-
cated butterﬂy network of size (cid:96) × n followed by a dense linear layer of size k × (cid:96), and obtain a
network with fewer parameters but loose very little in terms of representation. Hence, it is worth-
while investigating the representational power of the encoder-decoder butterﬂy network

Y = DEBX .
(1)
Here, X, Y and D are as in the encoder-decoder network, E ∈ Rk×(cid:96) is a dense matrix, and B is
an (cid:96) × n truncated butterﬂy network. In the encoder-decoder butterﬂy network the encoding is done
using EB, and decoding is done using D. This reduces the number of parameters in the encoding
matrix from kn (as in the encoder-decoder network) to k(cid:96) + O(n log (cid:96)). Again the objective is to
learn matrices D and E, and the truncated butterﬂy network B such that ||Y − Y ||2
F is minimized.
The optimal solution is denoted as Y ∗, D∗, E∗, and B∗. Theorem 1 shows that the loss at a critical
point of such a network depends on the eigenvalues of Σ(B) = Y X T BT (BXX T BT )−1XY T ,
when BXX T BT is invertible and Σ(B) has (cid:96) distinct positive eigenvalues.The loss L is deﬁned as
||Y − Y ||2
F.
Theorem 1. Let D, E and B be a point of the encoder-decoder network with a truncated butterﬂy
network satisfying the following: a) BXX T BT is invertible, b) Σ(B) has (cid:96) distinct positive eigen-
values λ1 > . . . > λ(cid:96), and c) the gradient of L(Y ) with respect to the parameters in D and E
matrix is zero. Then corresponding to this point (and hence corresponding to every critical point)
there is an I ⊆ [(cid:96)] such that L(Y ) at this point is equal to tr(Y Y T ) − (cid:80)
i∈I λi. Moreover if the
point is a local minima then I = [k].

The proof of Theorem 1 is given in Appendix C. We also compare our result with that of Baldi and
Hornik (1989) and Kawaguchi (2016), which study the optimization landscape of dense linear neural
networks in Appendix C. From Theorem 1 it follows that if B is ﬁxed and only D and E are trained
then a local minima is indeed a global minima. We use this to claim a worst-case guarantee using
a two-phase learning approach to train an auto-encoder. In this case the optimal solution is denoted
as Bk(Y ), DB, and EB. Observe that when Y = X, Bk(X) is the best rank k approximation of X
computed from the rows of BX.

Two phase learning for auto-encoder: Let (cid:96) = k log k + k/(cid:15) and consider a two phase learning
strategy for auto-encoders, as follows: In phase one B is sampled from an FJLT distribution, and
then only D and E are trained keeping B ﬁxed. Suppose the algorithm learns D(cid:48) and E(cid:48) at the end

5Possibly multiple D∗ and E∗ exist such that Y ∗ = D∗E∗X.

5

Under review as a conference paper at ICLR 2021

Dataset Name
Cifar-10 Krizhevsky (2012)
Cifar-10 Krizhevsky (2012)
Cifar-100 Krizhevsky (2012)
Imagenet Deng et al. (2009)
CoNLL-03 Tjong Kim Sang and De Meulder (2003)
CoNLL-03 Tjong Kim Sang and De Meulder (2003)
Penn Treebank (English) Marcus et al. (1993)

Task
Image classiﬁcation
Image classiﬁcation
Image classiﬁcation
Image classiﬁcation
Named Entity Recognition (English)
Named Entity Recognition (German)
Part-of-Speech Tagging

Model
EfﬁcientNet Tan and Le (2019)
PreActResNet18 He et al. (2016)
seresnet152 Hu et al. (2020)
senet154 Hu et al. (2020)
Flair’s Sequence Tagger Akbik et al. (2018) Akbik et al. (2019)
Flair’s Sequence Tagger Akbik et al. (2018) Akbik et al. (2019)
Flair’s Sequence Tagger Akbik et al. (2018) Akbik et al. (2019)

Table 1: Data and the corresponding architectures used in the fast matrix multiplication using but-
terﬂy matrices experiments.

of phase one, and X (cid:48) = D(cid:48)E(cid:48)B. Then Theorem 1 guarantees that, assuming Σ(B) has (cid:96) distinct
positive eigenvalues and D(cid:48), E(cid:48) are a local minima, D(cid:48) = DB, E(cid:48) = EB, and X (cid:48) = Bk(X).
Namely X (cid:48) is the best rank k approximation of X from the rows of BX. From Proposition 2 with
probability at least 1
2 , L(X (cid:48)) ≤ (1 + (cid:15))∆k. In the second phase all three matrices are trained to
improve the loss. In Sections 6.2 and 6.3 we train an encoder-decoder butterﬂy network using the
standard gradient descent method. In these experiments the truncated butterﬂy network is initialized
by sampling it from an FJLT distribution, and D and E are initialized randomly as in Pytorch.

6 EXPERIMENTS ON DENSE LAYER REPLACEMENT AND

ENCODER-DECODER BUTTERFLY NETWORK

In this section we report the experimental results based on the ideas presented in Sections 4.2 and 5.

6.1 REPLACING DENSE LINEAR LAYERS BY THE PROPOSED ARCHITECTURE

This experiment replaces a dense linear layer of size n2 × n1 in common deep learning architectures
with the network proposed in Section 4.2.6 The truncated butterﬂy networks are initialized by sam-
pling it from the FJLT distribution, and the dense matrices are initialized randomly as in Pytorch.
We set k1 = log n1 and k2 = log n2. The datasets and the corresponding architectures considered
are summarized in Table 1. For each dataset and model, the objective function is the same as de-
ﬁned in the model, and the generalization and convergence speed between the original model and
the modiﬁed one (called the butterﬂy model for convenience) are compared. Figure 7 in Appendix
D.1 reports the number of parameters in the dense linear layer of the original model, and in the
replaced network, and Figure 8 in Appendix D.1 displays the number of parameter in the original
model and the butterﬂy model. In particular, Figure 7 shows the signiﬁcant reduction in the number
of parameters obtained by the proposed replacement. On the left of Figure 1, the test accuracy of
the original model and the butterﬂy model is reported, where the black vertical lines denote the error
bars corresponding to standard deviation, and the values above the rectangles denote the average
accuracy. On the right of Figure 1 observe that the test accuracy for the butterﬂy model trained with
stochastic gradient descent is even better than the original model trained with Adam in the ﬁrst few
epochs. Figure 12 in Appendix D.1 compares the test accuracy in the the ﬁrst 20 epochs of the orig-
inal and butterﬂy model. The results for the NLP tasks in the interest of space are reported in Figure
9, Appendix D.1. The training and inference times required for the original model and the butterﬂy
model in each of these experiments are reported in Figures 10 and 11 in Appendix D.1. We remark
that the modiﬁed architecture is also trained for fewer epochs. In almost all the cases the modiﬁed
architecture does better than the normal architecture, both in the rate of convergence and in the ﬁnal
accuracy/F 1 score. Moreover, the training time for the modiﬁed architecture is less.

6.2 ENCODER-DECODER BUTTERFLY NETWORK WITH SYNTHETIC GAUSSIAN AND REAL

DATA

This experiment tests whether gradient descent based techniques can be used to train encoder-
In all the experiments in this section Y = X. Five types of data
decoder butterﬂy network.
matrices are tested, whose attributes are speciﬁed in Table 2.7 Two among them are random and

6In all the architectures considered the ﬁnal linear layer before the output layer is replaced, and n1 and n2

depend on the architecture.

7In Table 2 HS-SOD denotes a dataset for hyperspectral images from natural scenes (Imamoglu et al., 2018).

6

Under review as a conference paper at ICLR 2021

Figure 1: Left: comparison of ﬁnal test accuracy with different image classiﬁcation models and data
sets; Right: comparison of test accuracy in the ﬁrst few epochs with different models and optimizers
on CIFAR-10 with PreActResNet18

three are constructed using standard public real image datasets. In the interest of space, the con-
struction of the data matrices is explained in Appendix D.2. For the matrices constructed from the
image datasets, the input coordinates are randomly permuted, which ensures the network cannot
take advantage of the spatial structure in the data. For each of the data matrices the loss obtained
via training the truncated butterﬂy network with the Adam optimizer is compared to ∆k (denoted as
PCA) and ||Jk(X) − X||2
F where J is an (cid:96) × n matrix sampled from the FJLT distribution (denoted
as FJLT+PCA). Figure 2 reports the loss on Gaussian 1 and MNIST, whereas Figure 13 in Appendix
D.2 reports the loss for the remaining data matrices. Observe that for all values of k the loss for
the encoder-decoder butterﬂy network is almost equal to ∆k, and is in fact ∆k for small and large
values of k.

Name
Gaussian 1
Gaussian 2
MNIST
Olivetti
HS-SOD

n
1024
1024
1024
1024
1024

d
1024
1024
1024
4096
768

rank
32
64
1024
1024
768

Table 2: Data used in the truncated butterﬂy auto-encoder reconstruction experiments

Figure 2: Approximation error on data matrix with various methods for various values of k. Left:
Gaussian 1 data, Right: MNIST data

6.3 TWO-PHASE LEARNING FOR ENCODER-DECODER BUTTERFLY NETWORK

This experiment is similar to the experiment in Section 6.2 but the training in this case is done in two
phases. In the ﬁrst phase, B is ﬁxed and the network is trained to determine an optimal D and E. In
the second phase, the optimal D and E determined in phase one are used as the initialization, and the

7

Under review as a conference paper at ICLR 2021

network is trained over D, E and B to minimize the loss. Theorem 1 ensures worst-case guarantees
for this two phase training (see below the theorem). Figure 3 reports the approximation error of an
image from Imagenet. The red and green lines in Figure 3 correspond to the approximation error at
the end of phase one and two respectively.

Figure 3: Approximation error achieved by different methods and the same zoomed on in the right

7 SKETCHING ALGORITHM FOR LOW-RANK MATRIX DECOMPOSITION

PROBLEM USING BUTTERFLY NETWORK

The recent inﬂuential work by Indyk et al. (2019) considers a supervised learning approach to com-
pute an (cid:96)×n pre-conditioning matrix B for low-rank approximation of n×d matrices. The matrix B
has a ﬁxed sparse structure as in Clarkson and Woodruff (2009), each column as one non-zero entry
(chosen randomly) which are learned to minimize the loss over a training set of matrices. In this
section, we present experiments with the setting being similar to that in Indyk et al. (2019), except
that B is now represented as an (cid:96) × n truncated butterﬂy network. Our setting is similar to that in
Indyk et al. (2019), except that B is now represented as an (cid:96) × n truncated butterﬂy network. Our
experiments suggests that indeed a learned truncated butterﬂy network does better than a random
matrix, and even a learned B as in Indyk et al. (2019).
Setup: Suppose X1, . . . , Xt ∈ Rn×d are training matrices sampled from a distribution D. Then a
B is computed that minimizes the following empirical loss: (cid:80)
F. We compute
Bk(Xi) using truncated SVD of BXi (as in Algorithm 1, Indyk et al. (2019)). Similar to Indyk
et al. (2019), the matrix B is learned by the back-propagation algorithm that uses a differentiable
SVD implementation to calculate the gradients, followed by optimization with Adam such that the
butterﬂy structure of B is maintained. The learned B can be used as the pre-processing matrix for
any matrix in the future. The test error for a matrix B and a test set Te is deﬁned as follows:
(cid:2) ||X − Xk||2

(cid:3) − AppTe, where AppTe = EX∼Te

i∈[t] ||Xi − Bk(Xi)||2

(cid:2) ||X − Bk(X)||2

ErrTe(B) = EX∼Te

(cid:3) .

F

F

Experiments and Results: The experiments are performed on the datasets shown in Table 3. In
HS-SOD Imamoglu et al. (2018) and CIFAR-10 Krizhevsky (2012) 400 training matrices (t = 400),
and 100 test matrices are sampled, while in Tech 200 training matrices (t = 200), and 95 test
matrices are sampled. In Tech Davido et al. (2004) each matrix has 835,422 rows but on average
only 25,389 rows and 195 columns contain non-zero entries. For the same reason as in Section 6.2
in each dataset, the coordinates of each row are randomly permuted. Some of the matrices in the
datasets have much larger singular values than the others, and to avoid imbalance in the dataset,
the matrices are normalized so that their top singular values are all equal, as done in Indyk et al.
(2019). For each of the datasets, the test error for the learned B via our truncated butterﬂy structure

Name
HS-SOD 1
CIFAR-10
Tech

n
1024
32
25,389

d
768
32
195

Table 3: Data used in the Sketching algorithm for low-rank matrix decomposition experiments.

8

Under review as a conference paper at ICLR 2021

is compared to the test errors for the following three cases: 1) B is a learned as a sparse sketching
matrix as in Indyk et al. (2019), b) B is a random sketching matrix as in Clarkson and Woodruff
(2009), and c) B is an (cid:96) × n Gaussian matrix. Figure 4 compares the test error for (cid:96) = 20, and
k = 10, where AppTe = 10.56. Figure 14 in Appendix E compares the test errors of the different
methods in the extreme case when k = 1, and Figure 15 in Appendix E compares the test errors of
the different methods for various values of (cid:96). Table 4 in Appendix E in Appendix E reports the test
error for different values of (cid:96) and k. Figure 16 in in Appendix E shows the test error for (cid:96) = 20 and
k = 10 during the training phase on HS-SOD. In Figure 16 it is observed that the butterﬂy learned
is able to surpass sparse learned after a merely few iterations.

Figure 5 compares the test error for the learned B via our truncated butterﬂy structure to a learned
matrix B with N non-zero entries in each column – the N non-zero location for each column are
chosen uniformly at random. The reported test errors are on HS-SOD, when (cid:96) = 20 and k = 10.
Interestingly, the error for butterﬂy learned is not only less than the error for sparse learned (N = 1
as in (Indyk et al., 2019)) but also less than than the error for dense learned (N = 20). In particular,
our results indicate that using a learned butterﬂy sketch can signiﬁcantly reduce the approximation
loss compared to using a learned sparse sketching matrix.

Figure 4: Test error by different sketching
matrices on different data sets

8 DISCUSSION AND FUTURE WORK

Figure 5: Test errors for various values of N
and a learned butterﬂy matrix

Discussion: Among other things, this work showed that it is beneﬁcial to replace dense linear layer
in deep learning architectures with a more compact architecture (in terms of number of parameters),
using truncated butterﬂy networks. This approach is justiﬁed using ideas from efﬁcient matrix ap-
proximation theory from the last two decades. however, results in additional logarithmic depth to the
network. This issue raises the question of whether the extra depth may harm convergence of gradient
descent optimization. To start answering this question, we show, both empirically and theoretically,
that in linear encoder-decoder networks in which the encoding is done using a butterﬂy network,
this typically does not happen. To further demonstrate the utility of truncated butterﬂy networks, we
consider a supervised learning approach as in Indyk et al. (2019), where we learn how to derive low
rank approximations of a distribution of matrices by multiplying a pre-processing linear operator
represented as a butterﬂy network, with weights trained using a sample of the distribution.

Future Work: The main open questions arising from the work are related to better understanding the
optimization landscape of butterﬂy networks. The current tools for analysis of deep linear networks
do not apply for these structures, and more theory is necessary. It would be interesting to determine
whether replacing dense linear layers in any network, with butterﬂy networks as in Section 4.2 harms
the convergence of the original matrix. Another direction would be to check empirically whether
adding non-linear gates between the layers (logarithmically many) of a butterﬂy network improves
the performance of the network. In the experiments in Section 6.1, we have replaced a single dense
layer by our proposed architecture. It would be worthwhile to check whether replacing multiple
dense linear layers in the different architectures harms the ﬁnal accuracy. Similarly, it might be
insightful to replace a convolutional layer by an architecture based on truncated butterﬂy network.
Finally, since our proposed replacement reduces the number of parameters in the network, it might
be possible to empirically show that the new network is more resilient to over-ﬁtting.

9

Under review as a conference paper at ICLR 2021

ACKNOWLEDGEMENT

This project has received funding from European Union’s Horizon 2020 research and innovation
program under grant agreement No 682203 -ERC-[ Inf-Speed-Tradeoff].

REFERENCES

N. Ailon and B. Chazelle. The fast johnson–lindenstrauss transform and approximate nearest neigh-

bors. SIAM J. Comput., 39(1):302–322, 2009.

N. Ailon and E. Liberty. Fast dimension reduction using rademacher series on dual BCH codes.

Discret. Comput. Geom., 42(4):615–630, 2009.

A. Akbik, D. Blythe, and R. Vollgraf. Contextual string embeddings for sequence labeling.

In
COLING 2018, 27th International Conference on Computational Linguistics, pages 1638–1649,
2018.

A. Akbik, T. Bergmann, and R. Vollgraf. Pooled contextualized embeddings for named entity recog-
nition. In NAACL 2019, 2019 Annual Conference of the North American Chapter of the Associa-
tion for Computational Linguistics, page 724–728, 2019.

K. Alizadeh, P. Anish, F. Ali, and R. Mohammad. Butterﬂy transform: An efﬁcient fft based neu-
ral architecture design. In IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), June 2020.

P. Baldi and K. Hornik. Neural networks and principal component analysis: Learning from examples

without local minima. Neural Networks, 2(1):53–58, 1989.

A. L. Cambridge. The olivetti faces dataset, 1994.

Y. Cheng, F. X. Yu, R. S. Feris, S. Kumar, A. N. Choudhary, and S. Chang. An exploration of
parameter redundancy in deep networks with circulant projections. In 2015 IEEE International
Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pages 2857–
2865. IEEE Computer Society, 2015.

K. L. Clarkson and D. P. Woodruff. Numerical linear algebra in the streaming model. In M. Mitzen-
macher, editor, Proceedings of the 41st Annual ACM Symposium on Theory of Computing, STOC
2009, pages 205–214. ACM, 2009.

J. Cooley and J. Tukey. An algorithm for the machine calculation of complex fourier series. Mathe-

matics of Computation, 19(90):297–301, 1965.

T. Dao, A. Gu, M. Eichhorn, A. Rudra, and C. R´e. Learning fast algorithms for linear transforms
using butterﬂy factorizations. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the
36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach,
California, USA, volume 97 of Proceedings of Machine Learning Research, pages 1517–1527.
PMLR, 2019.

T. Dao, N. S. Sohoni, A. Gu, M. Eichhorn, A. Blonder, M. Leszczynski, A. Rudra, and C. R.
In 8th
’e. Kaleidoscope: An efﬁcient, learnable representation for all structured linear maps.
International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April
26-30, 2020, 2020.

D. Davido, E. Gabrilovich, and S. Markovitch. Parameterized generation of labeled datasets for
text categorization based on a hierarchical directory. In 27th Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval, SIGIR ’04, pages 250–257,
2004.

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical

Image Database. In CVPR09, 2009.

10

Under review as a conference paper at ICLR 2021

M. Denil, B. Shakibi, L. Dinh, M. Ranzato, and N. de Freitas. Predicting parameters in deep learn-
In Advances in Neural Information Processing Systems 26: 27th Annual Conference on

ing.
Neural Information Processing Systems 2013., pages 2148–2156, 2013.

C. Ding, S. Liao, Y. Wang, Z. Li, N. Liu, Y. Zhuo, C. Wang, X. Qian, Y. Bai, G. Yuan, X. Ma,
Y. Zhang, J. Tang, Q. Qiu, X. Lin, and B. Yuan. Circnn: accelerating and compressing deep neural
networks using block-circulant weight matrices. In Proceedings of the 50th Annual IEEE/ACM
International Symposium on Microarchitecture, MICRO 2017, pages 395–408. ACM, 2017.

K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In Computer
Vision - ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands, October 11-14,
2016, Proceedings, Part IV, volume 9908 of Lecture Notes in Computer Science, pages 630–645.
Springer, 2016.

J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu. Squeeze-and-excitation networks.

IEEE Trans.

Pattern Anal. Mach. Intell., 42(8):2011–2023, 2020.

N. Imamoglu, Y. Oishi, X. Zhang, Y. F. G. Ding, T. Kouyama, and R. Nakamura. Hyperspectral
image dataset for benchmarking on salient object detection. In Tenth International Conference on
Quality of Multimedia Experience, (QoMEX), pages 1–3, 2018.

P. Indyk, A. Vakilian, and Y. Yuan. Learning-based low-rank approximations. In H. M. Wallach,
H. Larochelle, A. Beygelzimer, F. d’Alch´e-Buc, E. B. Fox, and R. Garnett, editors, Advances in
Neural Information Processing Systems 32: Annual Conference on Neural Information Process-
ing Systems 2019, NeurIPS 2019, pages 7400–7410, 2019.

V. Jain, N. Pillai, and A. Smith. Kac meets johnson and lindenstrauss: a memory-optimal, fast

johnson-lindenstrauss transform. arXiv, 03 2020.

W. Johnson and J. Lindenstrauss. Extensions of lipschitz maps into a hilbert space. Contemporary

Mathematics, 26:189–206, 01 1984. doi: 10.1090/conm/026/737400.

K. Kawaguchi. Deep learning without poor local minima.

In Advances in Neural Information
Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016,
December 5-10, 2016, Barcelona, Spain, pages 586–594, 2016.

F. Krahmer and R. Ward. New and improved johnson–lindenstrauss embeddings via the restricted
isometry property. SIAM Journal on Mathematical Analysis, 43:1269–1281, 06 2011. doi: 10.
1137/100810447.

A. Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto, 2012.

Y. LeCun and C. Cortes. MNIST handwritten digit database, 2010.

N. Lee, T. Ajanthan, and P. H. S. Torr. Snip: single-shot network pruning based on connection sen-
sitivity. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans,
LA, USA, May 6-9, 2019. OpenReview.net, 2019.

Y. Li and H. Yang. Interpolative butterﬂy factorization. SIAM J. Scientiﬁc Computing, 39(2), 2017.

Y. Li, H. Yang, E. R. Martin, K. L. Ho, and L. Ying. Butterﬂy factorization. Multiscale Model.

Simul., 13(2):714–732, 2015.

Z. Lu, V. Sindhwani, and T. N. Sainath. Learning compact recurrent neural networks.

In 2016
IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2016, pages
5960–5964. IEEE, 2016.

M. P. Marcus, B. Santorini, and M. A. Marcinkiewicz. Building a large annotated corpus of English:
The Penn Treebank. Computational Linguistics, 19(2):313–330, 1993. URL https://www.
aclweb.org/anthology/J93-2004.

E. Michielssen and A. Boag. A multilevel matrix decomposition algorithm for analyzing scattering
from large structures. IEEE Transactions on Antennas and Propagation, 44(8):1086–1093, 1996.

11

Under review as a conference paper at ICLR 2021

D. C. Mocanu, E. Mocanu, P. Stone, P. H. Nguyen, M. Gibescu, and A. Liotta. Scalable training of
artiﬁcial neural networks with adaptive sparse connectivity inspired by network science. Nature
Communications, 9:2383, 2018. doi: 10.1038/s41467-018-04316-3.

M. Moczulski, M. Denil, J. Appleyard, and N. de Freitas. ACDC: A structured efﬁcient linear layer.
In Y. Bengio and Y. LeCun, editors, 4th International Conference on Learning Representations,
ICLR 2016, 2016.

M. O’Neil, F. Woolfe, and V. Rokhlin. An algorithm for the rapid evaluation of special function

transforms. Applied and Computational Harmonic Analysis, 28(2):203 – 226, 2010.

T. N. Sainath, B. Kingsbury, V. Sindhwani, E. Arisoy, and B. Ramabhadran. Low-rank matrix
In IEEE
factorization for deep neural network training with high-dimensional output targets.
International Conference on Acoustics, Speech and Signal Processing, ICASSP 2013, Vancouver,
BC, Canada, May 26-31, 2013, pages 6655–6659. IEEE, 2013.

T. Sarl´os. Improved approximation algorithms for large matrices via random projections. In 47th
Annual IEEE Symposium on Foundations of Computer Science (FOCS 2006), pages 143–152.
IEEE Computer Society, 2006.

D. S. Seljebotn. WAVEMOTH-FAST SPHERICAL HARMONIC TRANSFORMS BY BUTTER-
FLY MATRIX COMPRESSION. The Astrophysical Journal Supplement Series, 199(1):5, 2012.

V. Sindhwani, T. N. Sainath, and S. Kumar. Structured transforms for small-footprint deep learning.
In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neu-
ral Information Processing Systems 28: Annual Conference on Neural Information Processing
Systems 2015, pages 3088–3096, 2015.

M. Tan and Q. V. Le. Efﬁcientnet: Rethinking model scaling for convolutional neural networks. In
K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International Conference
on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of
Proceedings of Machine Learning Research, pages 6105–6114. PMLR, 2019.

A. T. Thomas, A. Gu, T. Dao, A. Rudra, and C. R´e. Learning compressed transforms with low dis-
placement rank. In Advances in Neural Information Processing Systems 31: Annual Conference
on Neural Information Processing Systems 2018, pages 9066–9078, 2018.

E. F. Tjong Kim Sang and F. De Meulder. Introduction to the CoNLL-2003 shared task: Language-
independent named entity recognition. In Proceedings of the Seventh Conference on Natural Lan-
guage Learning at HLT-NAACL 2003, pages 142–147, 2003. URL https://www.aclweb.
org/anthology/W03-0419.

S. Verdenius, M. Stol, and P. Forr´e. Pruning via iterative ranking of sensitivity statistics. CoRR,

abs/2006.00896, 2020.

C. Wang, G. Zhang, and R. B. Grosse. Picking winning tickets before training by preserving gradient
ﬂow. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa,
Ethiopia, April 26-30, 2020. OpenReview.net, 2020.

Z. Yang, M. Moczulski, M. Denil, N. de Freitas, A. J. Smola, L. Song, and Z. Wang. Deep fried
convnets. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, pages 1476–
1483. IEEE Computer Society, 2015.

J. Ye, L. Wang, G. Li, D. Chen, S. Zhe, X. Chu, and Z. Xu. Learning compact recurrent neural
networks with block-term tensor decomposition. In 2018 IEEE Conference on Computer Vision
and Pattern Recognition, CVPR 2018, pages 9378–9387. IEEE Computer Society, 2018.

12

Under review as a conference paper at ICLR 2021

A BUTTERFLY DIAGRAM FROM SECTION 1

Figure 6 referred to in the introduction is given here.

Figure 6: A 16 × 16 butterﬂy network represented as a 4-layered graph on the left, and as product
of 4 sparse matrices on the right. The white entries are the non-zero entries of the matrices.

B PROOF OF PROPOSITION 1

The proof of the proposition will use the following well known fact (Lemma B.1 below) about
FJLT (more generally, JL) distributions (see Ailon and Chazelle (2009); Ailon and Liberty (2009);
Krahmer and Ward (2011)).
Lemma B.1. Let x ∈ Rn be a unit vector, and let J ∈ Rk×n be a matrix drawn from an FJLT
distribution. Then for all (cid:15) < 1 with probability at least 1 − e−Ω(k(cid:15)2):

(cid:107)x − J T Jx(cid:107) ≤ (cid:15) .

(2)

By Lemma B.1 we have that with probability at least 1 − e−Ω(k1(cid:15)2),

Henceforth, we condition on the event (cid:107)x−J T
norm (cid:107)W (cid:107) of W :

(cid:107)x − J T

1 J1x(cid:107) ≤ (cid:15)(cid:107)x(cid:107) = (cid:15) .

(3)
1 J1x(cid:107) ≤ (cid:15)(cid:107)x(cid:107). Therefore, by the deﬁnition of spectral

(cid:107)W x − W J T
Now apply Lemma B.1 again on the vector W J T
bility at least 1 − e−Ω(k2(cid:15)2),

1 J1x(cid:107) ≤ (cid:15)(cid:107)W (cid:107) .
(4)
1 J1x and transformation J2 to get that with proba-

(cid:107)W J T

1 J1x − J T

2 J2W J T

1 J1x(cid:107) ≤ (cid:15)(cid:107)W J T

1 J1x(cid:107).

Henceforth, we condition on the event (cid:107)W J T
the last right hand side, we use the triangle inequality together with (4):

1 J1x − J T

2 J2W J T

1 J1x(cid:107) ≤ (cid:15)(cid:107)W J T

(5)
1 J1x(cid:107). To bound

Combining (5) and (6) gives:

(cid:107)W J T

1 J1x(cid:107) ≤ (cid:107)W x(cid:107) + (cid:15)(cid:107)W (cid:107) ≤ (cid:107)W (cid:107)(1 + (cid:15)).

(cid:107)W J T

1 J1x − J T

2 J2W J T

1 J1x(cid:107) ≤ (cid:15)(cid:107)W (cid:107)(1 + (cid:15)).

(6)

(7)

13

Under review as a conference paper at ICLR 2021

Finally,

(cid:107)J T

2 J2W J T

1 J1x − W x(cid:107) = (cid:107)(J T

1 J1x) + (W J T

1 J1x − W x)(cid:107)

1 J1x − W J T

2 J2W J T
≤ (cid:15)(cid:107)W (cid:107)(1 + (cid:15)) + (cid:15)(cid:107)W (cid:107)
= (cid:107)W (cid:107)(cid:15)(2 + (cid:15)) ≤ 3(cid:107)W (cid:107)(cid:15) ,

(8)

where the ﬁrst inequality is from the triangle inequality together with (4) and (7), and the second
inequality is from the bound on (cid:15). The proposition is obtained by adjusting the constants hiding
inside the Ω() notation in the exponent in the proposition statement.

C PROOF OF THEOREM 1

We ﬁrst note that our result continues to hold even if B in the theorem is replaced by any structured
matrix. For example the result continues to hold if B is an (cid:96) × n matrix with one non-zero entry per
column, as is the case with a random sparse sketching matrix Clarkson and Woodruff (2009). We
also compare our result with that Baldi and Hornik (1989); Kawaguchi (2016).

Comparison with Baldi and Hornik (1989) and Kawaguchi (2016): The critical points of the
encoder-decoder network are analyzed in Baldi and Hornik (1989). Suppose the eigenvalues of
Y X T (XX T )−1XY T are γ1 > . . . > γm > 0 and k ≤ m ≤ n. Then they show that corresponding
to a critical point there is an I ⊆ [m] such that the loss at this critical point is equal to tr(Y Y T ) −
(cid:80)
i∈I γi, and the critical point is a local/global minima if and only if I = [k]. Kawaguchi (2016)
later generalized this to prove that a local minima is a global minima for an arbitrary number of
hidden layers in a linear neural network if m ≤ n. Note that since (cid:96) ≤ n and m ≤ n in Theorem 1,
replacing X by BX in Baldi and Hornik (1989) or Kawaguchi (2016) does not imply Theorem 1 as
it is.
Next, we introduce a few notation before delving into the proof. Let r = (Y − Y )T , and vec(r) ∈
Rmd is the entries of r arranged as a vector in column-ﬁrst ordering, (∇vec(DT )L(Y ))T ∈ Rmk and
(∇vec(ET )L(Y ))T ∈ Rk(cid:96) denote the partial derivative of L(Y ) with respect to the parameters in
vec(DT ) and vec(ET ) respectively. Notice that ∇vec(DT )L(Y ) and ∇vec(ET )L(Y ) are row vectors
of size mk and k(cid:96) respectively. Also, let PD denote the projection matrix of D, and hence if D is a
matrix with full column-rank then PD = D(DT · D)−1 · DT . The n × n identity matrix is denoted
as In, and for convenience of notation let ˜X = B · X. First we prove the following lemma which
gives an expression for D and E if ∇vec(DT )L(Y ) and ∇vec(ET )L(Y ) are zero.
Lemma C.1 (Derivatives with respect to D and E).

1. ∇vec(DT )L(Y ) = vec(r)T (Im ⊗ (E · ˜X)T ), and

2. ∇vec(ET )L(X) = vec(r)T (D ⊗ ˜X)T

Proof.

1. Since L(Y ) = 1

2 vec(r)T · vec(r),

∇vec(DT )L(Y ) = vec(r)T · ∇vec(DT )vec(r) = vec(r)T (vec(DT )( ˜X T · ET · DT ))

= vec(r)T (Im ⊗ (E · ˜X)T ) · ∇vec(DT )vec(DT ) = vec(r)T (Im ⊗ (E · ˜X)T )

2. Similarly,

∇vec(ET )L(Y ) = vec(r)T · ∇vec(ET )vec(r) = vec(r)T (vec(ET )( ˜X T · ET · DT ))

= vec(r)T (D ⊗ ˜X T ) · ∇vec(ET )vec(ET ) = vec(r)T (D ⊗ ˜X T )

Assume the rank of D is equal to p. Hence there is an invertible matrix C ∈ Rk×k such that
˜D = D · C is such that the last k − p columns of ˜D are zero and the ﬁrst p columns of ˜D are linearly
independent (via Gauss elimination). Let ˜E = C −1 ·E. Without loss of generality it can be assumed
˜D ∈ Rd×p, and ˜E ∈ Rp×d, by restricting restricting ˜D to its ﬁrst p columns (as the remaining are

14

Under review as a conference paper at ICLR 2021

zero) and ˜E to its ﬁrst p rows. Hence, ˜D is a full column-rank matrix of rank p, and DE = ˜D ˜E.
Claims C.1 and C.2 aid us in the completing the proof of the theorem. First the proof of theorem is
completed using these claims, and at the end the two claims are proved.
Claim C.1 (Representation at the critical point).

1. ˜E = ( ˜DT ˜D)−1 ˜DT Y ˜X T ( ˜X · ˜X T )−1

2. ˜D ˜E = P ˜DY ˜X T ( ˜X · ˜X T )−1

Claim C.2.

1. ˜EB ˜D = ( ˜EBY ˜X T ˜ET )( ˜E ˜X ˜X T ˜ET )−1

2. P ˜DΣ = ΣP ˜D = P ˜DΣP ˜D

We denote Σ(B) as Σ for convenience. Since Σ is a real symmetric matrix, there is an orthogonal
matrix U consisting of the eigenvectors of Σ, such that Σ = U ∧ U T , where ∧ is a m × m
diagonal matrix whose ﬁrst (cid:96) diagonal entries are λ1, . . . , λ(cid:96) and the remaining entries are zero. Let
u1, . . . , um be the columns of U . Then for i ∈ [(cid:96)], ui is the eigenvector of Σ corresponding to the
eigenvalue λi, and {u(cid:96)+1, . . . , udy } are the eigenvectors of Σ corresponding to the eigenvalue 0.

Note that PU T ˜D = U T ˜D( ˜DT U T U ˜D)−1 ˜DT U = U T P ˜DU , and from part two of Claim C.2
we have

(U PU T ˜DU T )Σ = Σ(U PU T ˜DU T )
U · PU T ˜D ∧ U T = U ∧ PU T ˜DU T
PU T ˜D∧ = ∧PU T ˜D
Since PU T ˜D commutes with ∧, PU T ˜D is a block-diagonal matrix comprising of two blocks P1 and
P2: the ﬁrst block P1 is an (cid:96) × (cid:96) diagonal block, and P2 is a (m − (cid:96)) × (m − (cid:96)) matrix. Since
PU T ˜D is orthogonal projection matrix of rank p its eigenvalues are 1 with multiplicity p and 0 with
multiplicity m − p. Hence at most p diagonal entries of P1 are 1 and the remaining are 0. Finally
observe that

(10)
(11)

(9)

L(Y ) = tr((Y − Y )(Y − Y )T )

= tr(Y Y T ) − 2tr(Y Y T ) + tr(Y Y
= tr(Y Y T ) − 2tr(P ˜DΣ) + tr(P ˜DΣP ˜D)
= tr(Y Y T ) − tr(P ˜DΣ)

)

T

The second line in the above equation follows using the fact that tr(Y Y T ) = tr(Y Y
), the third
line in the above equation follows by substituting Y = P ˜DY ˜X T · ( ˜X · ˜X T )−1 · ˜X (from part two
of Claim C.1), and the last line follows from part two of Claim C.2. Substituting Σ = U ∧ U T , and
P ˜D = U PU T ˜DU T in the above equation we have,

T

L(Y ) = tr(Y Y T ) − tr(U PU T ˜D ∧ U T )
= tr(Y Y T ) − tr(PU T ˜D∧)
The last line the above equation follows from the fact that tr(U P ˜U T D ∧ U T ) = tr(PU T ˜D ∧ U T U ) =
tr(PU T ˜D∧). From the structure of PU T ˜D and ∧ it follows that there is a subset I ⊆ [(cid:96)], |I| ≤ p such
that tr(PU T ˜D∧) = (cid:80)

i∈I λi. Hence, L(Y ) = tr(Y Y T ) − (cid:80)

i∈I λi.

Since P ˜D = U PU T ˜DU T , there is a p × p invertible matrix M such that

˜D = (U · V )I (cid:48) · M , and ˜E = M −1(V T U T )I (cid:48)Y ˜X T ( ˜X ˜X T )−1
where V is a block-diagonal matrix consisting of two blocks V1 and V2: V1 is equal to I(cid:96), and V2 is
an (m − (cid:96)) × (m − (cid:96)) orthogonal matrix, and I (cid:48) is such that I ⊆ I (cid:48) and |I (cid:48)| = p. The relation for ˜E
in the above equation follows from part one of Claim C.1. Note that if I (cid:48) ⊆ [(cid:96)], then I = I (cid:48), that is
I consists of indices corresponding to eigenvectors of non-zero eigenvalues.

Recall that ˜D was obtained by truncating the last k − p zero rows of DC, where C was a

15

I (cid:48))T U (cid:48)

I (cid:48) = Ip. Deﬁne
D(cid:48) = U (cid:48)

Under review as a conference paper at ICLR 2021

k × k invertible matrix simulating the Gaussian elimination. Let [M |Op×(k−p)] denoted the p × k
matrix obtained by augmenting the columns of M with (k − p) zero columns. Then

Similarly, there is a p × (k − p) matrix N such that

D = (U V )I (cid:48)[M |Op×(k−p)]C −1 .

E = C[ M −1

N ]((U V )I (cid:48))T Y ˜X T ( ˜X ˜X T )−1
where [ M −1
N ] denotes the k × p matrix obtained by augmenting the rows of M −1 with the rows of
N . Now suppose I (cid:54)= [k], and hence I (cid:48) (cid:54)= [k]. Then we will show that there are matrices D(cid:48) and
E(cid:48) arbitrarily close to D and E respectively such that if Y (cid:48) = D(cid:48)E(cid:48) ˜X then L(Y (cid:48)) < L(Y ). There
is an a ∈ [k] \ I (cid:48), and b ∈ I (cid:48) such that λa > λb (λb could also be zero). Denote the columns of
the matrix U V as {v1, . . . , vm}, and observe that vi = ui for i ∈ [(cid:96)] (from the structure of V ). For
(cid:15) > 0 let u(cid:48)
2 (vb + (cid:15)ua). Deﬁne U (cid:48) as the matrix which is equal to U V except that
the column vector vb in U V is replaced by u(cid:48)
b in U (cid:48). Since a ∈ [k] ⊆ [(cid:96)] and a /∈ I (cid:48), va = ua and
(U (cid:48)

b = (1 + (cid:15)2)− 1

and let Y (cid:48) = D(cid:48)E(cid:48) ˜X. Now observe that, D(cid:48)E(cid:48) = U (cid:48)

I (cid:48)[M |Op×(k−p)]C −1 , and E(cid:48) = C[ M −1

N ](U (cid:48)

I (cid:48))T Y ˜X T ( ˜X ˜X T )−1
I (cid:48)(UI (cid:48))T Y ˜X T ( ˜X ˜X T )−1, and that

L(Y (cid:48)) = tr(Y Y T ) −

(cid:88)

i∈I

λi −

(cid:15)2

1 + (cid:15)2 (λa − λb) = L(Y ) −

(cid:15)2

1 + (cid:15)2 (λa − λb)

Since (cid:15) can be set arbitrarily close to zero, it can be concluded that there are points in the neigh-
bourhood of Y such that the loss at these points are less than L(Y ). Further, since L is convex with
respect to the parameters in D (respectively E), when the matrix E is ﬁxed (respectively D is ﬁxed)
Y is not a local maximum. Hence, if I (cid:54)= [k] then Y represents a saddle point, and in particular Y
is local/global minima if and only if I = [k].

Proof of Claim C.1. Since ∇vec(ET )L(X) is equal to zero, from the second part of Lemma C.1 the
following holds,

˜X(Y − Y )T D = ˜XY T D − ˜XY

T

D = 0

⇒ ˜X ˜X T ET DT D = ˜XY T D

Taking transpose on both sides

(12)
Substituting DE as ˜D ˜E in Equation 12, and multiplying Equation 12 by C T on both the sides from
the left, Equation 13 follows.

⇒ DT DE ˜X ˜X T = DT Y ˜X T

Since ˜D is full-rank, we have

and,

⇒ ˜DT ˜D ˜E ˜X ˜X T = ˜DT Y ˜X T

˜E = ( ˜DT ˜D)−1 ˜DT Y ˜X T ( ˜X ˜X T )−1.

˜D ˜E = P ˜DY ˜X T ( ˜X ˜X T )−1

(13)

(14)

(15)

Proof of Claim C.2. Since ∇vec(DT )L(Y ) is zero, from the ﬁrst part of Lemma C.1 the following
holds,

E ˜X(Y − Y )T = E ˜XY T − E ˜X · Y
⇒ E ˜X ˜X T ET DT = E ˜XY T
(16)
Substituting ET · DT as ˜ET · ˜DT in Equation 12, and multiplying Equation 16 by C −1 on both the
sides from the left Equation 17 follows.

= 0

T

˜E ˜X ˜X T ˜ET ˜DT = ˜E ˜XY T

(17)

16

Under review as a conference paper at ICLR 2021

Taking transpose of the above equation we have,

˜D ˜E ˜X ˜X T ˜ET = Y ˜X T ˜ET
(18)
From part 1 of Claim C.1, it follows that ˜E has full row-rank, and hence ˜E ˜X ˜X T ˜ET is invertible.
Multiplying the inverse of ˜E ˜X ˜X T ˜ET from the right on both sides and multiplying ˜EB from the
left on both sides of the above equation we have,

(19)
This proves part one of the claim. Moreover, multiplying Equation 18 by ˜DT from the right on both
sides

˜EB ˜D = ( ˜EBY ˜X T ˜ET )( ˜E ˜X ˜X T ˜ET )−1

˜D ˜E ˜X ˜X T ˜ET ˜DT = Y ˜X T ˜ET ˜DT

⇒ (P ˜DY ˜X T ( ˜X ˜X T )−1)( ˜X ˜X T )(( ˜X ˜X T )−1 ˜XY T P ˜D) = Y ˜X T (( ˜X ˜X T )−1 ˜XY T · P ˜D)

⇒ P ˜DY ˜X T ( ˜X ˜X T )−1 ˜XY T P ˜D = Y ˜X T ( ˜X ˜X T )−1 ˜XY T · P ˜D

The second line the above equation follows by substituting ˜D ˜E = P ˜DY ˜X T ( ˜X ˜X T )−1 (from part 2
of Claim C.1). Substituting Σ = Y ˜X T ( ˜X ˜X T )−1 ˜XY T in the above equation we have

P ˜DΣP ˜D = Σ · P ˜D
= P ˜D, and ΣT = Σ, we also have ΣP ˜D = P ˜DΣ.

Since P T
˜D

D ADDITIONAL TABLES AND PLOTS FROM SECTION 6

D.1 PLOTS FROM SECTION 6.1

Figure 7 displays the number of parameters in the dense linear layer of the original model and in
the replaced butterﬂy based network. Figure 9 reports the results for the NLP tasks done as part of
experiment in Section 6.1. Figure 8 displays the number of parameter in the original model and the
butterﬂy model. Figures 10 and 11 reports the training and inference times required for the original
model and the butterﬂy model in each of the experiments. The training and and inference times in
Figures 10 and 11 are averaged over 100 runs. Figure 12 is the same as the right part of Figure 1 but
here we compare the test accuracy of the original and butterﬂy model for the the ﬁrst 20 epochs.

Figure 7: Number of parameters in the dense linear layer of the original model and in the replaced
butterﬂy based architecture; Left: Vision data, Right: NLP

D.2 PLOTS FROM SECTION 6.2

Data Matrices: The data matrices are as in Table 2. Gaussian 1 and Gaussian 2 are Gaussian
matrices with rank 32 and 64 respectively. Rank r Gaussian matrices are constructed as follows: r
orthonormal vectors of size 1024 are sampled at random and the columns of the matrix are random
linear combinations of these vectors determined by choosing the coefﬁcients independently and
uniformly at random from the Gaussian distribution with mean 0 and variance 0.01. The data matrix
for MNIST is constructed as follows: each row corresponds to an image represented as a 28 ×

17

Under review as a conference paper at ICLR 2021

Figure 8: Total number of parameters in the original model and the butterﬂy model; Left: Vision
data, Right: NLP

Figure 9: Right: Final F1 Score for different NLP models and data sets. Left: F1 comparison in the
ﬁrst few epochs with different models on CoNLL-03 Named Entity Recognition (English) with the
ﬂair’s Sequence Tagger

Figure 10: Training/Inference times for Vision Data; Left: Training time, Right: Inference time

28 matrix (pixels) sampled uniformly at random from the MNIST database of handwritten digits
(LeCun and Cortes, 2010) which is extended to a 32 × 32 matrix by padding numbers close to zero
and then represented as a vector of size 1024 in column-ﬁrst ordering8. Similar to the MNIST every
row of the data matrix for Olivetti corresponds to an image represented as a 64 × 64 matrix sampled
uniformly at random from the Olivetti faces data set (Cambridge, 1994), which is represented as
a vector of size 4096 in column-ﬁrst ordering. Finally, for HS-SOD the data matrix is a 1024 ×
768 matrix sampled uniformly at random from HS-SOD – a dataset for hyperspectral images from
natural scenes (Imamoglu et al., 2018).

Figure 13 reports the losses for the Gaussian 2, Olivetti, and Hyper data matrices.

8Close to zero entries are sampled uniformly at random according to a Gaussian distribution with mean zero

and variance 0.01.

18

Under review as a conference paper at ICLR 2021

Figure 11: Training/Inference times for NLP; Left: Training time, Right: Inference time

Figure 12: Comparison of test accuracy in the ﬁrst 20 epochs with different models and optimizers
on CIFAR-10 with PreActResNet18

Figure 13: Approximation error on data matrix with various methods for various values of k. From
left to right: Gaussian 2, Olivetti, Hyper

E MISSING PLOTS FROM SECTION 7

In this section we state a few additional cases that were done as part of the experiment in Section 7.
Figure 14 compares the test errors of the different methods in the extreme case when k = 1. Figure
15 compares the test errors of the different methods for various values of (cid:96). Figure 16 shows the
test error for (cid:96) = 20 and k = 10 during the training phase on HS-SOD. Observe that the butterﬂy

19

Under review as a conference paper at ICLR 2021

learned is able to surpass sparse learned after a merely few iterations. Finally Table 4 compares the
test error for different values of (cid:96) and k.

Figure 14: Test errors on HS-SOD for (cid:96) = 20 and k = 1, zoomed on butterﬂy and sparse learned in
the right

Figure 15: Test error when k = 10, (cid:96) = [10, 20, 40, 60, 80] on HS-SOD, zoomed on butterﬂy and
sparse learned in the right

Figure 16: Test error when k = 10, (cid:96) = 20 during the training phase on HS-SOD

F BOUND ON NUMBER OF EFFECTIVE WEIGHTS IN TRUNCATED

BUTTERFLY NETWORK

A butterﬂy network for dimension n, which we assume for simplicity to be an integral power of 2,
is log n layers deep. Let p denote the integer log n. The set of nodes in the ﬁrst (input) layer will be
denoted here by V (0). They are connected to the set of n nodes V (1) from the next layer, and so on
until the nodes V (p) of the output layer. Between two consecutive layers V (i) and V (i+1), there are
2n weights, and each node in V (i) is adjacent to exactly two nodes from V (i+1).

When truncating the network, we discard all but some set S(p) ⊆ V (p) of at most (cid:96) nodes in the
last layer. These nodes are connected to a subset S(p−1) ⊆ V (p−1) of at most 2(cid:96) nodes from the

20

Under review as a conference paper at ICLR 2021

k, (cid:96), Sketch
1, 5, Butterﬂy
1, 5, Sparse
1, 5, Random
1, 10, Butterﬂy
1, 10, Sparse
1, 10, Random
10, 10, Butterﬂy
10, 10, Sparse
10, 10, Random
10, 20, Butterﬂy
10, 20, Sparse
10, 20, Random
10, 40, Butterﬂy
10, 40, Sparse
10, 40, Random
20, 20, Butterﬂy
20, 20, Sparse
20, 20, Random
20, 40, Butterﬂy
20, 40, Sparse
20, 40, Random
30, 30, Butterﬂy
30, 30, Sparse
30, 30, Random
30, 60, Butterﬂy
30, 60, Sparse
30, 60, Random

Hyper Cifar-10 Tech
0.188
0.173
0.0008
1.75
1.121
0.003
3.127
4.870
0.661
0.051
0.072
0.0002
0.455
0.671
0.002
1.44
1.82
0.131
0.619
0.751
0.031
7.154
6.989
0.489
18.805
26.133
5.712
0.568
0.470
0.012
3.134
3.122
0.139
8.22
2.097
9.216
0.111
0.006
0.991
0.081
3.304
0.544
1.38
0.058
8.14
0.229
15.268
4.173
0.703
0.024
3.441
0.247
6.848
1.334
1.25
0.027
7.519
0.749
13.168
3.486
0.409
0.014
2.993
0.331
5.124
2.105

Table 4: Test error for different (cid:96) and k

preceding layer using at most 2(cid:96) weights. By induction, for all i ≥ 0, the set of nodes S(p−i) ⊆
V (p−i) is of size at most 2i ·(cid:96), and is connected to the set S(p−i−1) ⊆ V (p−i−1) using at most 2i+1 ·(cid:96)
weights.

Now take k = (cid:100)log2(n/(cid:96))(cid:101). By the above, the total number of weights that can participate in a path
connecting some node in S(p) with some node in V (p−k) is at most

2(cid:96) + 4(cid:96) + · · · + 2k(cid:96) ≤ 4n .

From the other direction, the total number of weights that can participate in a path connecting any
node from V (0) with any node from V (p−k) is 2n times the number of layers in between, or more
precisely:

2n(p − k) = 2n(log2 n − (cid:100)log2(n/(cid:96))(cid:101)) ≤ 2n(log2 n − log2(n/(cid:96)) + 1) = 2n(log (cid:96) + 1) .

The total is 2n log (cid:96) + 6n, as required.

21

