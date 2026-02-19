Published as a conference paper at ICLR 2023

TENSOR-BASED SKETCHING METHOD FOR THE LOW-
RANK APPROXIMATION OF DATA STREAMS

Cuiyu Liu1, Chuanfu Xiao2 3, Mingshuo Ding1, Chao Yang2 3∗
1Academy for Advanced Interdisciplinary Studies, Peking University, Beijing, China
2School of Mathematical Sciences, Peking University, Beijing, China
3Changsha Institute for Computing and Digital Economy, Changsha, China
2101213203@stu.pku.edu.cn,
{chuanfuxiao,dingmingshuo,chao_yang}@pku.edu.cn

ABSTRACT

Low-rank approximation in data streams is a fundamental and significant task in
computing science, machine learning and statistics. Multiple streaming algorithms
have emerged over years and most of them are inspired by randomized algorithms,
more specifically, sketching methods. However, many algorithms are not able to
leverage information of data streams and consequently suffer from low accuracy.
Existing data-driven methods improve accuracy but the training cost is expensive
in practice. In this paper, from a subspace perspective, we propose a tensor-based
sketching method for low-rank approximation of data streams. The proposed
algorithm fully exploits the structure of data streams and obtains quasi-optimal
sketching matrices by performing tensor decomposition on training data. A series
of experiments are carried out and show that the proposed tensor-based method can
be more accurate and much faster than the previous work.

1

INTRODUCTION

There are many scenarios that require batch or real-time processing of data streams arising from,
e.g., video (Cyganek & Wo´zniak, 2017; Das, 2021), signal flow (Cichocki et al., 2015; Sidiropoulos
et al., 2017), hyperspectral images (Wang et al., 2017; Zhang et al., 2019) and numerical simulations
(Zhang et al., 2022; Larcher & Klein, 2019). A data stream can be seen as an ordered sequence of data
continuously generated from one or several distributions (Muthukrishnan, 2005; Indyk et al., 2019),
and the data per time slot can be usually represented as a matrix. Therefore, most of the processing
methods of data streams can be considered as operations on matrices, such as matrix multiplications,
linear system solutions and low-rank approximation. Wherein, low-rank matrix approximation plays
an important role in practical applications, such as independent component analysis (ICA) (Stone,
2002; Hyvärinen, 2013), principle component analysis (PCA) (Karamizadeh et al., 2020; Jolliffe &
Cadima, 2016), image denoising (Guo et al., 2015; Zhang et al., 2019).

In this work, we consider low-rank approximation of matrices from a data stream. Specifically, let
{Ad ∈ Rm×n}D
d=1 be matrices from a data stream D, then the low-rank approximation in D can be
described as:

min
Bd

∥Ad − Bd∥F , s.t. rank(Bd) ≤ r,

(1.1)

where d = 1, 2, · · · , D, ∥ · ∥F represents the Frobenius norm, and r ∈ Z+ is a user-specified target
rank.

Related work. A direct approach to solve problem 1.1 is to calculate the truncated rank-r singular
value decomposition (SVD) of Ad in turn, and the Eckart-Young theorem ensures that it is the
best low-rank approximation (Eckart & Young, 1936). However, it is too expensive to one by one
calculate the truncated rank-r SVD of Ad for all d = 1, 2, · · · , D, particularly when m or n is
large. To address this issue, many sketching algorithms have emerged such as the SCW algorithm
(Sarlos, 2006; Clarkson & Woodruff, 2009; 2017). Unfortunately, a notable weakness of sketching

∗Correspondence to Chao Yang.

1

Published as a conference paper at ICLR 2023

algorithms is that they achieve higher error than the best low-rank approximation, especially when
the sketching matrix is generated randomly from some distribution, such as Gaussian, Cauchy, or
Rademacher distribution (Indyk, 2006; Woolfe et al., 2008; Clarkson & Woodruff, 2009; Halko et al.,
2011; Clarkson & Woodruff, 2017). To improve accuracy, a natural idea is to perform a preprocessing
on the past data (seen as a training set) in order to better handle the future input matrices (seen as a
test set). This approach, which is often called the data-driven approach, has gained more attention
lately. For low-rank approximation, the pioneer of this work was (Indyk et al., 2019), who proposed a
learning-based method, that we henceforth refer to as IVY. In the IVY method, the sketching matrix
is set to be sparse, and the values of non-zero entries are learned instead of setting them randomly
as classical methods do. Specifically, learning is done by stochastic gradient descent (SGD), by
optimizing a loss function that portrays the quality of the low-rank approximation obtained by the
SCW algorithm as mentioned above. To improve accuracy, (Liu et al., 2020) followed the line of IVY
by additionally optimizing the location of the non-zero entries of the sketching matrix S, not only
their values. Recently, (Indyk et al., 2021) proposed a Few-Shot data-driven low-rank approximation
algorithm, and their motivation is to reduce the training time cost of (Indyk et al., 2019). Wherein,
they proposed an algorithm namely FewShotSGD by minimizing a new loss function that measures
the distance in subspace between the sketching matrix S and all left-SVD factor matrices of the
training matrices, with SGD. However, these data-driven approaches all involve learning mechanisms,
which require iterations during the optimization process. This raises a question: can we design an
efficient method, such as a non-iterative method, to get a better sketching matrix with both short
training time and high approximation quality? It would be an important step for the development of
data-driven methods, especially in scenarios requiring low latency.

Our contributions. In this work, we propose a new data-driven approach for low-rank approximation
of data streams, motivated by a subspace perspective. Specifically, we observe that a perfect sketching
matrix S ∈ Rk×m should be close to the top-k subspace of U d, where U d is the left-SVD factor
matrix of Ad. Due to the relevance of matrices in a data stream, it allows us to develop a new
sketching matrix S to approximate the top-k subspace of U d for all d = 1, · · · , D. Perhaps the
heavy learning mechanisms can be eliminated. In fact, our approach attains the sketching matrix
by minimizing a new loss function which is a relaxation of that in IVY. The most important thing
is that we can get the minimization of this loss function by tensor decomposition on the training
set, which is non-iterative. We refer to this method as tensor-based method. As an extension of the
main approach, we also develop the two-sided tensor-based algorithm, which involves two sketching
matrices S, W . These two sketching matrices can be obtained simultaneously by performing tensor
decomposition once. Both algorithms are significantly faster and more accurate than the previous
data-driven approaches.

2 PRELIMINARIES

The SCW algorithm. Randomized SVD is an efficient algorithm for computing the low-rank
approximation of matrices from a data stream. For example, the SCW algorithm, proposed by Sarlos,
Clarkson and Woodruff (Sarlos, 2006; Clarkson & Woodruff, 2009; 2017), is a classical randomized
SVD algorithm. The algorithm only computes the SVD of the compressed matrices SA and AV ,
and its time cost is O(r2(m + n)) when we set k = O(r). The detailed procedure is shown in
Algorithm 1.

Algorithm 1 The SCW algorithm (Sarlos, 2006; Clarkson & Woodruff, 2009; 2017).
Input: Matrix A ∈ Rm×n, sketching matrix S ∈ Rk×m, and target rank r < min{m, n}
1: ∼, ∼, V T ← full SVD of SA
2: [AV ]r ← truncated rank-r SVD of AV
3: ˆA ← [AV ]rV T
Output: Low-rank approximation of A: ˆA

In (Clarkson & Woodruff, 2009), it is proved that if S satisfies the property of Johnson-Lindenstrauss
Lemma, k = O(r log(1/δ)/ε) suffices the output ˆA to satisfy ∥A − ˆA∥F ≤ (1 + ε)∥A − [A]r∥F

2

Published as a conference paper at ICLR 2023

with probability 1−δ. Therefore, the approximation quality of the SCW algorithm is highly dependent
on the choice of the sketching matrix S. In general, the randomly generated sketching matrix does not
meet the accuracy requirements when we handle problems in a data stream, so can we design a new
S by utilizing the information of the data stream? This is the motivation of data-driven approaches.

The IVY algorithm. In (Indyk et al., 2019), the sketching matrix S is initialized by a sparse random
sign matrix as described in (Clarkson & Woodruff, 2009). The location of the non-zero entries is
fixed, while the values are optimized with SGD via the loss function as follow.

min
S∈Rk×m

(cid:88)

A∈Dtrain

∥A − SCW(A, S, r)∥2
F ,

(2.1)

where Dtrain is the training set sampled from the data stream D. This requires computing the gradient
of the SCW operator, which involves the SVD implementation (line 1 and 2 in Algorithm 1). IVY
uses a differential but inexact SVD based on the power method, and (Liu et al., 2020) suggested that
the SVD in PyTorch is also feasible and much more efficient.

The Few-Shot algorithm. In (Indyk et al., 2021), S is initialized the same way as IVY, and the
location of non-zero entries remains, too. The difference is that the authors optimize the non-zero
values by letting S to approximate the left top-r subspace of a few training matrices. Wherein, the
proposed algorithm namely FewShotSGD minimizes the following loss function:

min
S∈Rk×m

(cid:88)

U ∈Utrain

∥U T

r ST SU − I0∥2
F ,

(2.2)

where Utrain = {U : A = U ΣV T of all A ∈ Dtrain}, Ur denotes a matrix containing the first r
columns of U , and I0 ∈ Rr×n has zero entries except that (I0)i,i = 1 for i = 1, · · · , r.
As shown in (Indyk et al., 2021), the goal of FewShotSGD is to get the sketch which preserves the left
top-r subspace of all matrices A ∈ Dtrain well and meanwhile is orthogonal to their bottom-(n − r)
subspace. This raises a question: can we directly obtain a subspace that is close to the top-r subspace
of all As? The answer is yes! In this way, all matrices A ∈ Dtrain are required to be viewed as
a whole, i.e., a third-order tensor. For illustration, we introduce some basics about tensor before
presenting our method.
Tensor basics. For convenience, we only consider the third-order tensor A ∈ Rm×n×D, and Ai,j,d
i,j,d.
represents the (i, j, d)-th entry of A. The Frobenius norm of A is defined as ∥A∥F =

A2

(cid:114) (cid:80)
i,j,d

The mode-n (n = 1, 2, 3) matricization of A is to reshape it to a matrix A(n). For example, the
mode-1 matricization of A is A(1) ∈ Rm×nD satisfying (A(1))i,1+(j−1)n+(d−1)mn = Ai,j,d. The
1-mode product of A and a matrix S ∈ Rk×m is denoted as B = A×1 S ∈ Rk×n×D, which satisfies
Ai,j,dSs,i. Tucker decomposition (Tucker, 1966) is one format of tensor decomposition,
Bs,j,d =

m
(cid:80)
i=1

which is also called higher-order singular value decomposition (HOSVD) (Lathauwer et al., 2000). It
decomposes a tensor into a set of factor matrices and one small core tensor of the same order. For
A ∈ Rm×n×D, its Tucker decomposition is

A = G ×1 U ×2 V ×3 W ,

where U ∈ Rm×r1, V ∈ Rn×r2, W ∈ RD×r3 are the column orthogonal factor matrices, G ∈
Rr1×r2×r3 is the core tensor, and (r1, r2, r3) is called the multilinear-rank of A. There are two
important variations of Tucker decomposition, i.e., Tucker1 and Tucker2 (Kolda & Bader, 2009) (1 or
2 modes of A are decomposed), which can be represented as A = G ×1 U and A = G ×1 U ×2 V ,
respectively.

3

TENSOR-BASED SKETCHING METHOD

In this section, we present our idea and method for low-rank approximation in data streams. The
goal is to employ the given training set to get the sketch S, inspired by IVY (Indyk et al., 2019) and
FewShotSGD (Indyk et al., 2021).

3

Published as a conference paper at ICLR 2023

3.1

TENSOR-BASED ALGORITHM

Our main algorithm, the tensor-based algorithm, is also a data-driven algorithm for low-rank ap-
proximation in data streams. Instead of minimizing the loss 2.1 in IVY, we consider a different loss,
motivated by a subspace perspective. This loss function is easier to optimize than 2.1 since to get its
minimization, only a Tucker1 decomposition is required, without learning mechanisms.
Let Dtrain = {Ad′ ∈ Rm×n}D′

d′=1, and the loss function we consider is

min
S∈Rk×m

(cid:88)

Ad′ ∈Dtrain

∥Ad′ − ST SAd′∥2

F , s.t. SST = Ik.

(3.1)

Using the row-wise orthogonality of S, we have ∥Ad′ − ST SAd′∥2
A ∈ Rm×n×D′
solve

F . Let
be a third-order tensor satisfying A:,:,d′ = Ad′. To minimize 3.1, it is equivalent to

F − ∥SAd′∥2

F = ∥Ad′∥2

∥SAd′∥2

F ⇐⇒ max

S∈Rk×m

∥SA(1)∥2

F , s.t. SST = Ik,

(3.2)

max
S∈Rk×m

(cid:88)

Ad′ ∈Dtrain

where A(1) = [A1|A2| · · · |AD′] is the mode-1 matricization of A. Further, as shown in (Kolda &
Bader, 2009), problem 3.2 is equivalent to

min
S∈Rk×m

∥A − G ×1 ST ∥2
F

s.t. G ∈ Rk×n×D′

, SST = Ik.

(3.3)

This is a Tucker1 decomposition of A along mode-1. Let A(1) = U (1)Σ(1)(V (1))T be the SVD
of A(1). The optimal sketch S∗ for problem 3.3 is (U (1))T
k , where (U (1))k is a matrix composed
of the first k columns in U (1) (refer to (Kolda & Bader, 2009)). We use the optimal S∗ as input
of SCW, and get the output of SCW as the low-rank approximation. The tensor-based algorithm is
summarized in Algorithm 2.

The motivation behind this choice of loss function is the theorem below, which illustrates the
relationship between our loss function 3.1 and that in IVY.
Theorem 1. Let Ad′ ∈ Rm×n be a matrix from the training set, and A ∈ Rm×n×D′
be a third-order
tensor satisfying A:,:,d′ = Ad′. Given the target rank r ∈ Z+, and a row-wise orthogonal matrix
S ∈ Rk×m, for any positive integer k > r, we have

D′
(cid:88)

d′=1

∥Ad′ − SCW(Ad′, S, r)∥2

F ≤ ∥A∥2

F − ∥[SA(1)]r∥2
F ,

(3.4)

where A(1) = [A1|A2| · · · |AD′] is the mode-1 matricization of A. Furthermore, with this relaxation,
problem 2.1 can be converted to our proposed problem 3.3.

Theorem 1 justifies the rationality of our choice of the loss function. Below we give an analysis that
using the sketch obtained by problem 3.3, the SCW computes a good low-rank approximation of A.

Analysis. In fact, our idea is similar to that in (Indyk et al., 2021) — both choosing the sketch S
to approximate the top-r row subspace of matrices in Dtrain. Let U ΣV T be the SVD of A, where
A is a matrix in Dtrain. Since there is strong relevance among matrices in Dtrain, it makes sense to
assume that S obtained by 3.3 is close in space to Uk (k > r), where Uk is a matrix composed of the
first k columns of U . In a special case where all matrices in Dtrain are the same, i.e., Ad′ = A for
d′ = 1, · · · , D′, using S obtained by 3.3, we have ∥UkU T
F = 0. Theorem 2 shows that
using S computed by tensor-based algorithm, the SCW gives a good low-rank approximation of A in
a data stream.
Theorem 2. Let U ΣV T be the SVD of A ∈ Rm×n, and Uk be a matrix composed of the first k
columns of U . Given a row-wise orthogonal sketching matrix S ∈ Rk×m satisfying ∥UkU T
k −
ST S∥2

k − ST S∥2

F < ε, then we have

∥A − SCW(A, S, r)∥2

F − ∥A − [A]r∥2

F < O(ε)∥A∥2
F .

(3.5)

4

Published as a conference paper at ICLR 2023

The proofs of Theorem 1 and 2 are provided fully in the Appendix.

Algorithm 2 The tensor-based algorithm for low-rank approximation of the data stream D.
Input: Test matrix A, training set {Ad′ ∈ Rm×n}D′

d′=1, rank r ≤ min{m, n}, # rows of the

sketching matrix k.

1: Tensorization: A ∈ Rm×n×D′
2: S ← Tucker1 decomposition of A along the mode-1
3: ˆA ← SCW(A, S, r)
Output: Low-rank approximation of A: ˆA

← {Ad′}D′

d′=1

3.2 TWO-SIDED TENSOR-BASED ALGORITHM

The two-sided tensor-based algorithm is an extension of the tensor-based algorithm in Section 3.1.
The motivation is that if we compute the Tucker2 decomposition of A mentioned in Theorem 1, two
sketching matrices S and W , would be computed at once. This means that besides using S for row
space compression, we can use W to compress the column space of A, too. To be clear, we consider

min
S∈Rk×m,W ∈Rl×n

∥A − G ×1 ST ×2 W T ∥2
F

s.t. G ∈ Rk×l×D′
SST = Ik, W W T = Il.

,

(3.6)

Unlike 3.3, the exact solution of problem 3.6 has no explicit form, but can be efficiently approximated
by an alternating iteration algorithm, namely higher-order orthogonal iteration (HOOI) (Lathauwer
et al., 2000; Kolda & Bader, 2009). We present the HOOI algorithm in the Appendix. However, the
SCW algorithm requires only one sketching matrix for computing low-rank approximation. As a
result, a new sketching algorithm for two sketches is required.

Two-sided SCW. To this end, we develop a new algorithm for low-rank approximation based on the
SCW algorithm, which we call two-sided SCW. It is worth mentioning that the full SVD in line 1 of
Algorithm 1 is used for orthogonalization, thus it can be replaced with QR decomposition to improve
the computational efficiency. With this in mind, the procedure of the two-sided SCW that we design
is as shown in Algorithm 3.

Algorithm 3 The two-sided SCW algorithm.
Input: Matrix A ∈ Rm×n, sketching matrices S ∈ Rk×m and W ∈ Rl×n, and target rank

r < min{m, n}

1: Q, ∼ ← QR decomposition of AT ST
2: P , ∼ ← QR decomposition of AW T
3: [P T AQ]r ← truncated rank-r SVD of P T AQ
4: P [P T AQ]rQT ← low-rank approximation of A
Output: Low-rank approximation of A: ˆA

Clearly, Algorithm 3 is more efficient than the original SCW when m, n are both large. The truncated
SVD only needs to be done on P T AQ ∈ Rl×k, which is much smaller in size than AV ∈ Rm×k in
Algorithm 1 (m > l).

The procedure of the two-sided tensor-based algorithm is similar to the previously introduced tensor-
based algorithm. First, reshape the training matrices to a third-order tensor A. Then, obtain two
sketching matrices S, W by computing the Tucker2 decomposition of A. Finally, taking S, W
and a test matrix A as input, use two-sided SCW to get the low-rank approximation of A. We
summarize this in Algorithm 4. Recall that we compute the Tucker2 decomposition of A by HOOI

5

Published as a conference paper at ICLR 2023

(Lathauwer et al., 2000). If k, l ∼ O(r), the time cost for Tucker2 decomposition with HOOI
is O(rmnD′ + r(m + n)D′2), while Tucker1 decomposition costs O(mn2D′). In addition, as
mentioned before, two-sided SCW is more efficient than the SCW algorithm. That means, the time
complexity of the two-sided algorithm is asymptotic less than the original tensor-based algorithm
because r ≪ m, n, usually. However, since two-sided SCW uses S, W to compress both the row
and column space of A while the SCW compresses the row space only, there would be some loss in
accuracy for the two-sided tensor-based algorithm compared to the tensor-based one.

Algorithm 4 The two-sided tensor-based algorithm for low-rank approximation of the data stream D.
Input: Test matrix A, training set {Ad′ ∈ Rm×n}D′

d′=1, target rank r ≤ min{m, n}, # rows of the

sketching matrix k and l.
1: Tensorization: A ∈ Rm×n×D′
2: S, W ← Tucker2 decomposition of A along the mode-1 and 2
3: ˆA ← Two-sided SCW(A, S, W , r)
Output: Low-rank approximation of A: ˆA

← {Ad′}D′

d′=1

4 NUMERICAL EXPERIMENTS

In this section, we test our algorithms and compare them to the existing data-driven algorithms for
low-rank approximation of data streams. We use three datasets for comparison — HSI (Imamoglu
et al., 2018), Logo (Indyk et al., 2019) and MRI.

Table 1: Summary of datasets used for experiment.

Name
HSI1
Logo2
MRI3

Description

Dimension

#Train

#Test

Hyper spectral images
Video
Magnetic resonance imaging

1024 × 768
3240 × 1920
217 × 181

100
100
30

400
400
120

We measure the quality of the sketching matrix S by the error on the test set, and the test error
is defined as Error = 1
, where Aopt is the best rank-r ap-
proximation of A, and ˆA is the low-rank approximation computed by the tested algorithms. In all
experiments, we set the rank r to 10, and the sketching size k = l = 20. Experiments are run on a
server equipped with an NVIDIA Tesla V100 card.

∥A− ˆA∥F −∥A−Aopt∥F
∥A−Aopt∥F

A∈Dtest

|Dtest|

(cid:80)

Baselines. As baselines, three methods are included — IVY (Indyk et al., 2019), Few-Shot (Indyk
et al., 2021), and Butterfly (Ailon et al., 2021).

IVY. As described in Ref. (Indyk et al., 2019), the sketching matrix is initialized by a sign matrix.
Its non-zero values are optimized by stochastic gradient descent (SGD) (Saad, 1998), which is an
iterative optimization method widely used in machine learning.

Few-Shot. In (Indyk et al., 2021), as IVY does, the sketching matrix is sparse, and the location
of the non-zero entries is fixed. The non-zero values of the sketch are also optimized by SGD.
They proposed one-shot closed-form algorithms (including 1Shot1Vec+IVY and 1Shot2Vec), and the
FewShotSGD algorithm with either 2 or 3 randomly chosen training matrices (i.e., FewShotSGD-2
and FewShotSGD-3). We compare our algorithms with all of them.

Butterfly.
In (Ailon et al., 2021), it is proposed to replace a dense linear layer in a neural net-
work by the butterfly network. They suggested using a butterfly gadget for learning the low-rank

1Retrieved from https://github.com/gistairc/HS-SOD.
2Retrieved from http://youtu.be/L5HQoFIaT4I.
3Retrieved from https://brainweb.bic.mni.mcgill.ca/cgi/brainweb2.

6

Published as a conference paper at ICLR 2023

Figure 1: Test error per training time with the target rank r = 10 and the sketching size k = 20. In
(d), SOTA represents the lowest test error that the baselines achieve.

approximation, also learning the non-zero values of a sparse sketching matrix by SGD, similarly to
IVY.

Since the baselines above all use one sketching matrix only, we compare our tensor-based algorithm
with them. For the two-sided tensor-based algorithm, we test its performance later in this section,
only comparing it with our tensor-based algorithm.

Training time and test error. We compare the test error per training time for each approach. The
results are reported in Figure 1. Table in (d) in Figure 1 lists the test error of the tensor-based
algorithm and the lowest test error among the baselines. The tensor-based algorithm achieves at least
0.55/0.64/0.27 times lower test error on HSI/Logo/MRI than the baselines. For the baselines,
the training matrices are required to be normalized to avoid the imbalance in the dataset before
the training starts. On HSI/Logo/MRI, this pre-processing of data takes 46.55s/447.67s/0.72s.
After that, the training for the sketching matrix could start. However, for our algorithms, this pre-
processing time can be avoided, because our algorithms are to compute the top-r subspace of the
training matrices which remains when the training data scales by a constant. On HSI/Logo/MRI,
the tensor-based algorithm takes 0.53s/4.76s/0.23s for training, which is much faster than IVY,
1Shot1Vec+IVY and Butterfly. As a result, our algorithm significantly outperforms the baselines —
much more accurate and faster.

Testing time. Next we report running time for the testing process on all datasets. Note that the
sketching matrix by the tensor-based algorithm is dense, while that of the baselines is sparse. This
results in the difference in the testing process, mainly on the matrix multiplication SA in the SCW
procedure. The baselines have approximately the same testing time since their sketching matrix
has the same sparsity. For the tensor-based algorithm, we use the built-in functionality in PyTorch,
i.e., torch.matmul(S, A), to compute SA, which provides great acceleration. On HSI/Logo/MRI,
the testing process for the tensor-based algorithm takes 0.52s/0.69s/0.28s. For the baselines,
the sketching matrix S is sparse and S is stored using two vectors — one for storing the location
of non-zero entries, and the other for storing the values of the non-zero entries. Suppose that the
location and value vectors are l, v. To compute SA, we can update SA[li, j] = SA[li, j] + viAij.
In this way, the testing time for the baselines is 19.31s/60.15s/1.71s on HSI/Logo/MRI, which is
much longer than that of the tensor-based algorithm, mainly because there is no software acceleration

7

0.060.080.0100.0120.0140.0160.0180.0Time (s)0.000.050.100.150.200.25Average test error(a) HSI//0.0500.0600.0700.0800.0900.0Time (s)0.000.050.100.150.200.25(b) Logo//05101520253035Time (s)0.000.050.100.150.200.25(c) MRIHSILogoMRISOTA0.03620.01680.0561Ours0.01980.01050.0149(d) Table: average test error1Shot1Vec+IVYButterflyIVYTensor-based (ours)1Shot2VecFewShotSGD-2FewShotSGD-3Published as a conference paper at ICLR 2023

Figure 2: Test error per training time compared with the dense baselines. Note that the training of the
dense baselines is faster than the original baselines for the use of torch.matmul() to compute SA.

applied here. However, the testing process for the tensor-based algorithm is accelerated by the use of
the built-in functionality torch.matmul() in PyTorch. If we restore the sparse S in the baselines as
the COO format and use torch.sparse.mm(S, A) to compute SA when testing, the testing time for
the baselines on HSI/Logo/MRI is reduced to 0.88s/1.17s/0.40s.

In view of the dense structure of our sketching matrix, we implement the baselines as dense ones and
compare our algorithm with them, including IVY (dense), 1Shot1Vec+IVY (dense), FewShotSGD-2
(dense) and FewShotSGD-3 (dense). The dense baselines optimize all entries of the sketching matrix,
not only the non-zero entries as the original baselines did. We show the results in Figure 2. The
test error for the tensor-based algorithm is 0.0198/0.0105/0.0149 on HSI/Logo/MRI, while the
lowest test error that the dense baselines achieve is 0.0191/0.0122/0.0124 on HSI/Logo/MRI.
With comparable accuracy, our approach has much shorter training time than the dense baselines.

Experiments for the two-sided algorithm. Finally, we test the performance of the two-sided tensor-
based algorithm. This algorithm uses two sketching matrices S, W for computing the low-rank
approximation. Table 2 shows the test error, the training time and the testing time for the two proposed
algorithms, the tensor-based algorithm and two-sided version. The tensor-based algorithm achieves
0.29/0.73/0.63 times lower test error on HSI/Logo/MRI than the two-sided algorithm. However,
the two-sided algorithm has both shorter training time and testing time. These results confirm our
analysis in Section 3.

Table 2: Test error, training time and testing time of the tensor-based algorithm and the two-sided
tensor-based algorithm.

Datasets

Algorithms

test error

training time (s)

testing time (s)

HSI

Logo

MRI

tensor-based

two-sided

tensor-based

two-sided

tensor-based

two-sided

0.020

0.069

0.011

0.015

0.015

0.024

0.53

0.39

4.76

1.36

0.23

0.17

0.52

0.41

0.69

0.50

0.28

0.12

100

100+400 = 30

100+400 = 100

Additional experiments. In the experiments above, we only evaluate the algorithms when the sample
ratio for training is 20% (
30+120 = 20% for HSI/Logo/MRI, respectively),
which we denote as sample_ratio = 20%. Figure 3 shows the performance of our proposed
algorithms under different values of sample_ratio, including 2%, 20% and 80%. The results show
that using only a small number of training matrices (sample_ratio = 2% for example), our algorithms
achieve low enough error. When sample_ratio increases from 2% to 80%, the test error of the tensor-
based algorithm decreases by a multiplicative factor of 0.952/0.917/0.684 on HSI/Logo/MRI,
and for the two-sided tensor-based algorithm, the corresponding factor is 0.873/0.789/0.606
on HSI/Logo/MRI. On MRI, the test error decreases more when the number of training matrices
increases compared to the other two datasets. In our opinion, this is because there is less stronger

8

0.048.050.052.054.056.058.060.062.0Time (s)0.000.050.100.150.200.25Average test error(a) HSI//0.0450.0460.0470.0480.0490.0500.0Time (s)0.000.050.100.150.200.25(b) Logo//012345Time (s)0.000.050.100.150.200.250.300.35(c) MRI1Shot1Vec+IVY(dense)IVY(dense)FewShotSGD-2(dense)FewShotSGD-3(dense)Tensor-based (ours)Published as a conference paper at ICLR 2023

Figure 3: Test error of the tensor-based algorithm and the two-sided tensor-based algorithm under
different number of training samples.

relevance among matrices of MRI. In general, increasing training samples improves the accuracy, but
not significantly.

For our approach, one of the limitations is that it is required to load the whole training tensor at
once. But for the baselines, the sketching matrix is learned by SGD and one of its advantages is
that only a few (batch-size) training matrices are required to load in memory at a time. However,
the results in Figure 3 show that a small number of training matrices are enough to achieve good
low-rank approximation for both the tensor-based algorithm and the two-sided tensor-based algorithm.
As a result, the memory usage of the proposed algorithms is also relatively low, comparable to the
baselines.

5 CONCLUSIONS AND FUTURE WORK

In this work, we propose an efficient and accurate approach to deal with low-rank approximation of
data streams, namely the tensor-based sketching method. From a subspace perspective, we develop a
tensor-based algorithm as well as a two-sided tensor-based algorithm. Numerical experiments show
that the two-sided tensor-based algorithm is faster but attains higher test error than the tensor-based
algorithm. Compared to the baselines, both algorithms are not only more accurate, but also far more
efficient.

This work mainly focuses on reducing the training time for generating the sketching matrix. However,
reducing the testing time is also of great interest. One of the approaches is to develop pass-efficient
sketching-based algorithms for low-rank approximation. In applications, the pass-efficiency becomes
crucial when the data size exceeds memory available in RAM. Further, in addition to low-rank
approximation, the idea of the tensor-based sketching method can be applied to more operations such
as ε-approximation and linear system solutions on data streams. We leave them for future work.

ACKNOWLEDGMENTS

The work was supported by the High-performance Computing Platform of Peking University. The
authors acknowledge it for supporting the computational work sincerely.

REFERENCES

N. Ailon, O. Leibovitch, and V. Nair. Sparse linear networks with a fixed butterfly structure: theory

and practice. In Uncertainty in Artificial Intelligence, pp. 1174–1184. PMLR, 2021.

D. Carlson. Minimax and interlacing thoerems for matrices. Linear Algebra and its Applications, 54:

153–172, 1983.

A. Cichocki, D. Mandic, L. De Lathauwer, G. Zhou, Q. Zhao, C. Caiafa, and H. A. Phan. Tensor
decompositions for signal processing applications: From two-way to multiway component analysis.
IEEE Signal Processing Magazine, 32(2):145–163, 2015.

K. L. Clarkson and D. P. Woodruff. Numerical linear algebra in the streaming model. In Proceedings

of the forty-first annual ACM symposium on Theory of computing, pp. 205–214, 2009.

9

HSILogoMRI0.000.020.040.060.08Average test error(a) Tensor-basedHSILogoMRI0.000.020.040.060.08(b) Two-sided tensor-basedsample_ratio=2%sample_ratio=20%sample_ratio=80%Published as a conference paper at ICLR 2023

K. L. Clarkson and D. P. Woodruff. Low-rank approximation and regression in input sparsity time.

Journal of the ACM (JACM), 63(6):1–45, 2017.

B. Cyganek and M. Wo´zniak. Tensor-based shot boundary detection in video streams. New Generation

Computing, 35(4):311–340, 2017.

S. Das. Hyperspectral image, video compression using sparse Tucker tensor decomposition. IET

Image Processing, 15(4):964–973, 2021.

C. Eckart and G. Young. The approximation of one matrix by another of lower rank. Psychometrika,

1(3):211–218, 1936.

Q. Guo, C. Zhang, Y. Zhang, and H. Liu. An efficient SVD-based method for image denoising. IEEE

Transactions on Circuits and Systems for Video Technology, 26(5):868–880, 2015.

N. Halko, P. G. Martinsson, and J. A. Tropp. Finding structure with randomness: Probabilistic
algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288,
2011.

A. Hyvärinen. Independent component analysis: Recent advances. Philosophical Transactions of the
Royal Society A: Mathematical, Physical and Engineering Sciences, 371(1984):20110534, 2013.

N. Imamoglu, Y. Oishi, X. Zhang, G. Ding, Y. Fang, T. Kouyama, and R. Nakamura. Hyperspectral
image dataset for benchmarking on salient object detection. In 2018 Tenth International Conference
on Quality of Multimedia Experience (qoMEX), pp. 1–3. IEEE, 2018.

P. Indyk. Stable distributions, pseudorandom generators, embeddings, and data stream computation.

Journal of the ACM (JACM), 53(3):307–323, 2006.

P. Indyk, A. Vakilian, and Y. Yuan. Learning-based low-rank approximations. In Proceedings of the
33rd International Conference on Neural Information Processing Systems, pp. 7402–7412, 2019.

P. Indyk, T. Wagner, and D. P. Woodruff. Few-shot data-driven algorithms for low rank approximation.

Advances in Neural Information Processing Systems, 34:10678–10690, 2021.

I. T. Jolliffe and J. Cadima. Principal component analysis: A review and recent developments.
Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering
Sciences, 374(2065):20150202, 2016.

S. Karamizadeh, S. M. Abdullah, A. A. Manaf, M. Zamani, and A. Hooman. An overview of principal

component analysis. Journal of Signal and Information Processing, 4, 2020.

T. G. Kolda and B. W. Bader. Tensor decompositions and applications. SIAM Rev., 51(3):455–500,

2009.

T. Von Larcher and R. Klein. Approximating turbulent and non-turbulent events with the tensor train

decomposition method. In Turbulent Cascades II, pp. 283–291. Springer, 2019.

L. De Lathauwer, B. De Moor, and J. Vandewalle. A multilinear singular value decomposition. SIAM

J. Matrix Anal. Appl., 21(4):1253–1278, 2000.

L. De Lathauwer, B. De Moor, and J. Vandewalle. On the best rank-1 and rank-(R1, R2, · · · , RN )

approximation of higher-order tensors. SIAM J. Matrix Anal. Appl., 21:1324–1342, 2000.

S. Liu, T. Liu, A. Vakilian, Y. Wan, and D. P. Woodruff. Learning the positions in countsketch. arXiv

preprint arXiv:2007.09890, 2020.

S. Muthukrishnan. Data streams: Algorithms and applications. Now Publishers Inc, 2005.

D. Saad. Online algorithms and stochastic approximations. Online Learning, 5:6–3, 1998.

R. Sarlos. Improved approximation algorithms for large matrices via random projections. In 2006
47th Annual IEEE Symposium on Foundations of Computer Science (FOCS’06), pp. 143–152.
IEEE, 2006.

10

Published as a conference paper at ICLR 2023

N. D. Sidiropoulos, L. De Lathauwer, X. Fu, K. Huang, E. E. Papalexakis, and C. Faloutsos.
Tensor decomposition for signal processing and machine learning. IEEE Transactions on Signal
Processing, 65(13):3551–3582, 2017.

J. V. Stone. Independent component analysis: An introduction. Trends in cognitive sciences, 6(2):

59–64, 2002.

L. R. Tucker. Some mathematical notes on three-mode factor analysis. Psychometrika, 31:279–311,

1966.

Y. Wang, J. Peng, Q. Zhao, Y. Leung, X. Zhao, and D. Meng. Hyperspectral image restoration via
total variation regularized low-rank tensor decomposition. IEEE Journal of Selected Topics in
Applied Earth Observations and Remote Sensing, 11(4):1227–1243, 2017.

F. Woolfe, E. Liberty, V. Rokhlin, and M. Tygert. A fast randomized algorithm for the approximation

of matrices. Applied and Computational Harmonic Analysis, 25(3):335–366, 2008.

G. Zhang, X. Zheng, S. Liu, M. Chen, C. Wang, and X. Wang. Three-dimensional wind velocity
reconstruction based on tensor decomposition and CFD data with experimental verification. Energy
Conversion and Management, 256:115322, 2022.

H. Zhang, L. Liu, W. He, and L. Zhang. Hyperspectral image denoising with total variation
regularization and nonlocal low-rank tensor decomposition. IEEE Transactions on Geoscience
and Remote Sensing, 58(5):3071–3084, 2019.

A APPENDIX

A.1 PROOF OF THEOREM 1

Proof. The inequality in 3.4 will be proved if we prove the following two inequalities.

D′
(cid:88)

d′=1

∥Ad′ − SCW(S, Ad′)∥2

F ≤

D′
(cid:88)

d′=1

∥Ad′ − ST [SAd′]r∥2
F ,

(A.1)

and

D′
(cid:88)

d′=1

∥Ad′ − ST [SAd′]r∥2

F ≤ ∥A∥2

F − ∥[SA(1)]r∥2
F .

(A.2)

First, we consider the inequality in A.1. Let Q ∈ Rn×k be a column-wise orthogonal matrix in the
row space of SAd′. By definition of SCW, we have

∥Ad′ − SCW(S, Ad′)∥2

F = ∥Ad′ − [Ad′Q]rQT ∥2

F = ∥Ad′∥2

F − ∥[Ad′Q]r∥2
F .

Similarly,

F = ∥Ad′∥2
Combing A.3 and A.4, A.1 follows immediately if we show

∥Ad′ − ST [SAd′]r∥2

F − ∥[SAd′]r∥2
F .

∥[SAd′]r∥F ≤ ∥[Ad′Q]r∥F .

(A.3)

(A.4)

(A.5)

Noting that

SAd′Q = U ΣV T Q = U ΣQ′,
where U ΣV T is the singular value decomposition of SAd′ and Q′ = V T Q. Since V and Q lie in
the same row space and are both column-wise orthogonal, it is easy to see that Q′ is a k-dimensional
orthogonal matrix. Thus, SAd′ and SAd′Q share the same singular values. Combining Cauchy
interlace theorem (Carlson, 1983), we have

∥[SAd′]r∥F = ∥[SAd′Q]r∥F ≤ ∥[Ad′Q]r∥F ,

(A.6)

which proves A.5.

11

Published as a conference paper at ICLR 2023

We now turn to the inequality in A.2. For convenience, we rewritten SA(1) and [SA(1)]r with block
components as

and

SA(1) = [SA1|SA2| · · · |SAD′],

[SA(1)]r = [B1|B2| · · · |BD′],

where Bi ∈ Rk×n for i = 1, · · · , D′. We then have

D′
(cid:88)

d′=1

∥SAd′∥2

F −

D′
(cid:88)

d′=1

∥[SAd′]r∥2

F =

≤

D′
(cid:88)

d′=1
D′
(cid:88)

d′=1

∥SAd′ − [SAd′]r∥2
F

∥SAd′ − Bd′∥2
F

= ∥SA(1)∥2

F − ∥[SA(1)]r∥2
F ,

where the Eckart-Young theorem is applied. It follows that

D′
(cid:88)

d′=1

∥[SAd′]r∥2

F ≥ ∥[SA(1)]r∥2
F ,

which is equivalent to in A.2.
Hence, we have proved that ∥A∥2

F − ∥[SA(1)]r∥2

F is a relaxation of (cid:80)

∥Ad′ −

F . Therefore, the problem 2.1 can be converted to minimize ∥A∥2

SCW(S, Ad′)∥2
i.e., maximize ∥[SA(1)]r∥2
maximizing ∥[SA(1)]r∥2
F is maximizing ∥SA(1)∥2
of optimizing problem 2.1, we can covert it to our proposed problem 3.3.

Ad′ ∈Dtrain
F − ∥[SA(1)]r∥2
F ,
F . Due to k > r, it is not difficult to verify that a sufficient condition for
F , which is equivalent to 3.3. As a result, instead

Hence, our proof is completed.

A.2 PROOF OF THEOREM 2

□

Proof. Let ˆU ˆΣ ˆV T be the SVD of the matrix SA. Using the definition of the SCW algorithm, we
have SCW(A, S, r) = [A ˆV ]r ˆV T . Further, since ˆV is column-wise orthogonal, we have

∥A − SCW(A, S, r)∥2

F = ∥A − [A ˆV ]r ˆV T ∥2

F = ∥A∥2

F − ∥[A ˆV ]r∥2
F .

Similarly, we have

∥A − [A]r∥2

F = ∥A∥2

F − ∥[U T

k A]r∥2
F .

Recall that U ΣV T is the SVD of A, and Uk be a matrix composed of the first k columns of U .
Based on the result A.6 in the proof of Theorem 1, we immediately get

Thus, we have

∥[A ˆV ]r∥2

F ≥ ∥[SA]r∥2
F .

∥A − SCW(A, S, r)∥2

F − ∥A − [A]r∥2

F − ∥[SA]r∥2
F

k A]r∥2
F ≤ ∥[U T
≤ ∥U T
k A∥2
= tr(AT (UkU T
≤ ∥UkU T
k − ST S∥2
k − ST S∥2
≤ ∥UkU T
≤ O(ε) ∥A∥2
F .

F − ∥SA∥2
F
k − ST S)A)
2 ∥A∥2
F
F ∥A∥2
F

Hence, our proof is completed.

12

(A.7)

(A.8)

□

Published as a conference paper at ICLR 2023

A.3 HOOI ALGORITHM

Algorithm 5 HOOI algorithm Lathauwer et al. (2000); Kolda & Bader (2009)

Input:

Tensor A ∈ RI1×I2···×IN
Truncation (R1, R2, · · · , RN )
Initial guess {U (n)

0

: n = 1, 2, · · · , N }

Output:

Low multilinear-rank approximation ˆA = G ×1 U (1) ×2 U (2) · · · ×N U (N )

)T ×n+1 (U (n+1)

k

k+1

for all n ∈ {1, 2, · · · , N } do

1: k ← 0
2: while not convergent do
3:
4:
5:
6:
7:
8:
9:
end for
10:
11: end while
12: G ← ΣV T in tensor format

k+1)T · · · ×n−1 (U (n−1)

B ← A ×1 (U (1)
B(n) ← B in matrix format
U , Σ, V T ← truncated rank-Rn SVD of B(n)
U (n)
Gk,n ← ΣV T in tensor format
k ← k + 1

k+1 ← U

)T · · · ×N (U (N )

k

)T

13

