Under review as a conference paper at ICLR 2022

LAYER-WISE ADAPTIVE MODEL AGGREGATION FOR
SCALABLE FEDERATED LEARNING

Anonymous authors
Paper under double-blind review

ABSTRACT

In Federated Learning, a common approach for aggregating local models across
clients is periodic averaging of the full model parameters. It is, however, known
that different layers of neural networks can have a different degree of model dis-
crepancy across the clients. The conventional full aggregation scheme does not
consider such a difference and synchronizes the whole model parameters at once,
resulting in inefﬁcient network bandwidth consumption. Aggregating the parame-
ters that are similar across the clients does not make meaningful training progress
while increasing the communication cost. We propose FedLAMA, a layer-wise
model aggregation scheme for scalable Federated Learning. FedLAMA adap-
tively adjusts the aggregation interval in a layer-wise manner, jointly considering
the model discrepancy and the communication cost. The layer-wise aggregation
method enables to ﬁnely control the aggregation interval to relax the aggregation
frequency without a signiﬁcant impact on the model accuracy. Our empirical study
shows that FedLAMA reduces the communication cost by up to 60% for IID data
and 70% for non-IID data while achieving a comparable accuracy to FedAvg.

1

INTRODUCTION

In Federated Learning, periodic full model aggregation is the most common approach for aggregat-
ing local models across clients. Many Federated Learning algorithms, such as FedAvg (McMahan
et al. (2017)), FedProx (Li et al. (2018)), FedNova (Wang et al. (2020)), and SCAFFOLD (Karim-
ireddy et al. (2020)), assume the underlying periodic full aggregation scheme. However, it has been
observed that the magnitude of gradients can be signiﬁcantly different across the layers of neural net-
works (You et al. (2019)). That is, all the layers can have a different degree of model discrepancy.
The periodic full aggregation scheme does not consider such a difference and synchronizes the en-
tire model parameters at once. Aggregating the parameters that are similar across all the clients does
not make meaningful training progress while increasing the communication cost. Considering the
limited network bandwidth in usual Federated Learning environments, such an inefﬁcient network
bandwidth consumption can signiﬁcantly harm the scalability of Federated Learning applications.

Many researchers have put much effort into addressing the expensive communication issue. Adap-
tive model aggregation methods adjust the aggregation interval to reduce the total communication
cost (Wang & Joshi (2018); Haddadpour et al. (2019)). Gradient (model) compression (Alistarh
et al. (2018); Albasyoni et al. (2020)), sparsiﬁcation (Wangni et al. (2017); Wang et al. (2018)),
low-rank approximation (Vogels et al. (2020); Wang et al. (2021)), and quantization (Alistarh et al.
(2017); Wen et al. (2017); Reisizadeh et al. (2020)) techniques directly reduce the local data size.
Employing heterogeneous model architectures across clients is also a communication-efﬁcient ap-
proach (Diao et al. (2020)). While all these works effectively tackle the expensive communication
issue from different angles, they commonly assume the underlying periodic full model aggregation.

To break such a convention of periodic full model aggregation, we propose FedLAMA, a novel layer-
wise adaptive model aggregation scheme for scalable and accurate Federated Learning. FedLAMA
ﬁrst prioritizes all the layers based on their contributions to the total model discrepancy. We present
a metric for estimating the layer-wise degree of model discrepancy at run-time. The aggregation
intervals are adjusted based on the layer-wise model discrepancy such that the layers with a smaller
degree of model discrepancy is assigned with a longer aggregation interval than the other layers.
The above steps are repeatedly performed once the entire model is synchronized once.

1

Under review as a conference paper at ICLR 2022

Our focus is on how to relax the model aggregation frequency at each layer, jointly considering the
communication efﬁciency and the impact on the convergence properties of federated optimization.
By adjusting the aggregation interval based on the layer-wise model discrepancy, the local mod-
els can be effectively synchronized while reducing the number of communications at each layer.
The model accuracy is marginally affected since the intervals are increased only at the layers that
have the smallest contribution to the total model discrepancy. Our empirical study demonstrates
that FedLAMA automatically ﬁnds the interval settings that make a practical trade-off between the
communication cost and the model accuracy. We also provide a theoretical convergence analysis of
FedLAMA for smooth and non-convex problems under non-IID data settings.

We evaluate the performance of FedLAMA across three representative image classiﬁcation bench-
mark datasets: CIFAR-10 (Krizhevsky et al. (2009)), CIFAR-100, and Federated Extended MNIST
(Cohen et al. (2017)). Our experimental results deliver novel insights on how to aggregate the local
models efﬁciently consuming the network bandwidth. Given a ﬁxed number of training iterations, as
the aggregation interval increases, FedLAMA reduces the communication cost by up to 60% under
IID data settings and 70% under non-IID data settings, while having only a marginal accuracy drop.

2 RELATED WORKS

Compression Methods – The communication-efﬁcient global model update methods can be catego-
rized into two groups: structured update and sketched update (Koneˇcn`y et al. (2016)). The structured
update indicates the methods that enforce a pre-deﬁned ﬁxed structure of the local updates, such as
low-rank approximation and random mask methods. The sketched update indicates the methods
that post-process the local updates via compression, sparsiﬁcation, or quantization. Both directions
are well studied and have shown successful results (Alistarh et al. (2018); Albasyoni et al. (2020);
Wangni et al. (2017); Wang et al. (2018); Vogels et al. (2020); Wang et al. (2021); Alistarh et al.
(2017); Wen et al. (2017); Reisizadeh et al. (2020)). The common principle behind these methods is
that the local updates can be replaced with a different data representation with a smaller size.

These compression methods can be independently applied to our layer-wise aggregation scheme
such that the each layer’s local update is compressed before being aggregated. Since our focus is on
adjusting the aggregation frequency rather than changing the data representation, we do not directly
compare the performance between these two approaches. We leave harmonizing the layer-wise
aggregation scheme and a variety of compression methods as a promising future work.

Similarity Scores – Canonical Correlation Analysis (CCA) methods are proposed to estimate the
representational similarity across different models (Raghu et al. (2017); Morcos et al. (2018)). Cen-
tered Kernel Alignment (CKA) is an improved extension of CCA (Kornblith et al. (2019)). While
these methods effectively quantify the degree of similarity, they commonly require expensive com-
putations. For example, SVCCA performs singular vector decomposition of the model and CKA
computes Hilbert-Schmidt Independence Criterion multiple times (Gretton et al. (2005)). In addi-
tion, the representational similarity does not deliver any information regarding the gradient differ-
ence that is strongly related to the convergence property. We will propose a practical metric for
estimating the layer-wise model discrepancy, which is cheap enough to be used at run-time.

Layer-wise Model Freezing – Layer freezing (dropping) is the representative layer-wise technique
for neural network training (Brock et al. (2017); Kumar et al. (2019); Zhang & He (2020); Goutam
et al. (2020)). All these methods commonly stop updating the parameters of the layers in a bottom-
up direction. These empirical techniques are supported by the analysis presented in (Raghu et al.
(2017)). Since the layers converge from the input-side sequentially, the layer-wise freezing can re-
duce the training time without strongly affecting the accuracy. These previous works clearly demon-
strate the advantages of processing individual layers separately.

3 BACKGROUND

Federated Optimization – We consider federated optimization problems as follows.

(cid:34)

min
x∈Rd

F (x) :=

(cid:35)

piFi(x)

,

m
(cid:88)

i=1

2

(1)

Under review as a conference paper at ICLR 2022

Algorithm 1: FedLAMA: Federated Layer-wise Adaptive Model Aggregation.
Input: τ (cid:48): base aggregation interval, φ: interval increasing factor, pi, i ∈ {1, · · · , m}.

1 τl ← τ (cid:48), ∀l ∈ {1, · · · , L};
2 for k = 1 to K do
SGD step: xi
3
for l = 1 to L do

k = xi

4

k−1 − η∇f (wi

k−1, ξk);

5

6

7

8

9

if k mod τl is 0 then

Synchronize layer l: u(l,k) ← (cid:80)m
(l,k)(cid:107)2(cid:17)
dl ← (cid:80)m

pi(cid:107)u(l,k) − xi

i=1

(cid:16)

i=1 pixi

(l,k);

/(τl(dim(u(l,k))) ;

if k mod φτ (cid:48) is 0 then

Adjust aggregation interval at all L layers (Algorithm 2).;

10 Output: uK;

where pi = ni/n is the ratio of local data to the total dataset, and Fi(x) = 1
ni
local objective function of client i. n is the global dataset size and ni is the local dataset size.

ξ∈D fi(x, ξ) is the

(cid:80)

FedAvg is a basic algorithm that solves the above minimization problem. As the degree of data het-
erogeneity increases, FedAvg converges more slowly. Several variants of FedAvg, such as FedProx,
FedNova, and SCAFFOLD, tackle the data heterogeneity issue. All these algorithms commonly
aggregate the local solutions using the periodic full aggregation scheme.

Model Discrepancy – All local SGD-based algorithms allow the clients to independently train their
local models within each communication round. The variance of stochastic gradients and heteroge-
neous data distribution can lead the local models to different directions on parameter space during
the local update steps. We formally deﬁne such a discrepancy among the models as follows.

m
(cid:88)

i=1

pi(cid:107)u − xi(cid:107)2,

where m is the number of clients, u is the synchronized model, and xi is client i’s local model.
This quantity bounds the difference between the local gradients and the global gradients under a
smoothness assumption on objective functions.

4 LAYER-WISE ADAPTIVE MODEL AGGREGATION

Layer Prioritization – In theoretical analysis, it is common to assume the smoothness of objective
functions such that the difference between local gradients and global gradients is bounded by a
scaled difference of the corresponding sets of parameters. Motivated by this convention, we deﬁne
‘layer-wise unit model discrepancy’, a useful metric for prioritizing the layers as follows.

dl =

(cid:80)m

i=1 pi(cid:107)ul − xi
τl(dim(ul))

l(cid:107)2

,

l ∈ {1, · · · , L}

(2)

where L is the number of layers, l is the layer index, u is the global parameters, xi is the client i’s
local parameters, τ is the aggregation interval, and dim(·) is the number of parameters.

This metric quantiﬁes how much each parameter contributes to the model discrepancy at each iter-
ation. The communication cost is proportional to the number of parameters. Thus, (cid:80)m
i=1 pi(cid:107)ul −
xi
l(cid:107)2/dim(ul) shows how much model discrepancy can be eliminated by synchronizing the layer
at a unit communication cost. This metric allows prioritizing the layers such that the layers with a
smaller dl value has a lower priority than the others.

Adaptive Model Aggregation Algorithm – We propose FedLAMA, a layer-wise adaptive model
aggregation scheme. Algorithm 1 shows FedLAMA algorithm. There are two input parameters: τ (cid:48)
is the base aggregation interval and φ is the interval increase factor. First, the parameters at layer
l are synchronized across the clients after every τl iterations (line 6). Then, the proposed metric

3

Under review as a conference paper at ICLR 2022

Algorithm 2: Layer-wise Adaptive Interval Adjustment.
Input: d: the observed model discrepancy at all L layers, τ (cid:48): the base aggregation interval, φ:

the interval increasing factor.

l=1 dim(ul);

1 Sorted model discrepancy: ˆd ← sort (d);
2 Sorted index of the layers: ˆi ← argsort (d);
3 Total model size: λ ← (cid:80)L
4 Total model discrepancy: δ ← (cid:80)L
5 for l = 1 to L do
δl ← ((cid:80)l
λl ← ((cid:80)l
Find the layer index: i ← ˆil ;
if δl < λl then
τi ← φτ (cid:48);

i=1
i=1 dim(ui))/λ;

ˆdi ∗ dim(ui))/δ;

9

7

8

6

10

l=1 dl ∗ dim(ul);

11

12

else

τi ← τ (cid:48);

13 Output: τ : the adjusted aggregation intervals at all L layers.;

dl is calculated using the synchronized parameters ul (line 7). At the end of every φτ (cid:48) iterations,
FedLAMA adjusts the model aggregation interval at all the L layers. (line 9).

Algorithm 2 ﬁnds the layers that can be less frequently aggregated making a minimal impact on the
total model discrepancy. First, the layer-wise degree of model discrepancy is estimated as follows.

δl =

(cid:80)l

(cid:80)L

i=1

i=1

ˆdi ∗ dim(ui)
ˆdi ∗ dim(ui)

,

(3)

where ˆdi is the ith smallest element in the sorted list of the proposed metric d. Given l layers with
the smallest dl values, δl quantiﬁes their contribution to the total model discrepancy. Second, the
communication cost impact is estimated as follows.

λl =

(cid:80)l

(cid:80)L

i=1 dim(ui)
i=1 dim(ui)

(4)

λl is the ratio of the parameters at the l layers with the smallest dl values. Thus, 1 − λl estimates
the number of parameters that will be more frequently synchronized than the others. As l increases,
δl increases while 1 − λl decreases monotonically. Algorithm 2 loops over the L layers ﬁnding the
l value that makes δl and 1 − λl similar. In this way, it ﬁnds the aggregation interval setting that
slightly sacriﬁces the model discrepancy while remarkably reducing the communication cost.

Figure 1 shows the δl and 1 − λl curves collected from a) CIFAR-10 (ResNet20) training and b)
CIFAR-100 (Wide-ResNet28-10) training. The x-axis is the number of layers to increase the aggre-
gation interval and the y-axis is the δl and 1 − λl values. The cross point of the two curves is much
lower than 0.5 on y-axis in both charts. For instance, in Figure 1.a), the two curves are crossed when
x value is 9, and the corresponding y value is near 0.2. That is, when the aggregation interval is
increased at those 9 layers, 20% of the total model discrepancy will increase by a factor of φ while
80% of the total communication cost will decrease by the same factor. Note that the cross points are
below 0.5 since the δl and 1 − λl are calculated using the dl values sorted in an increasing order.

It is worth noting that FedLAMA can be easily extended to improve the convergence rate at the cost
of having minor extra communications. In this work, we do not consider ﬁnding such interval set-
tings because it can increase the latency cost, which is not desired in Federated Learning. However,
in the environments where the latency cost can be ignored, such as high-performance computing
platforms, FedLAMA can accelerate the convergence by adjusting the intervals based on the cross
point of 1 − δl and λl calculated using the list of dl values sorted in a decreasing order.

Impact of Aggregation Interval Increasing Factor φ – In Federated Learning, the communication
latency cost is usually not negligible, and the total number of communications strongly affects the

4

Under review as a conference paper at ICLR 2022

Figure 1: The comparison between the model discrepancy increase factor δl and the communication
cost decrease factor 1 − λl for a) CIFAR-10 and b) CIFAR-100 training.

scalability. When increasing the aggregation interval, Algorithm 2 multiplies a pre-deﬁned small
constant φ to the ﬁxed base interval τ (cid:48) (line 10). This approach ensures that the communication
latency cost is not increased while the network bandwidth consumption is reduced by a factor of φ.

FedAvg can be considered as a special case of FedLAMA where φ is set to 1. When φ > 1, Fed-
LAMA less frequently synchronize a subset of layers, and it results in reducing their communication
costs. When increasing the aggregation interval, FedLAMA multiplies φ to the base interval τ (cid:48). So,
it is guaranteed that the whole model parameters are fully synchronized after φτ (cid:48) iterations. Because
of the layers with the base aggregation interval τ (cid:48), the total model discrepancy of FedLAMA after
φτ (cid:48) iterations is always smaller than that of FedAvg with an aggregation interval of φτ (cid:48).

5 CONVERGENCE ANALYSIS

5.1 PRELIMINARIES

Notations – All vectors in this paper are column vectors. x ∈ Rd denotes the parameters of one
local model and m is the number of clients. The stochastic gradient computed from a single training
data point ξ is denoted by g(x, ξ). For convenience, we use g(x) instead. The full batch gradient is
denoted by ∇F (x). We use (cid:107)·(cid:107) and (cid:107)·(cid:107)op to denote l2 norm and matrix operator norm, respectively.

Assumptions – We analyze the convergence rate of FedLAMA under the following assumptions.

1. (Smoothness). Each local objective function is L-smooth, that is, (cid:107)∇Fi(x) − ∇Fi(y)(cid:107) ≤

L(cid:107)x − y(cid:107), ∀i ∈ {1, · · · , m}.

2. (Unbiased Gradient). The stochastic gradient at each client is an unbiased estimator of the

local full-batch gradient: Eξ[gi(x, ξ)] = ∇Fi(x).

3. (Bounded Variance). The stochastic gradient at each client has bounded variance:

Eξ[(cid:107)gi(x, ξ) − ∇Fi(x)(cid:107)2 ≤ σ2], ∀i ∈ {1, · · · , m}, σ2 ≥ 0.
4. (Bounded Dissimilarity). For any sets of weights {pi ≥ 0}m

constants β2 ≥ 1 and κ2 ≥ 0 such that (cid:80)m
If local objective functions are identical to each other, β2 = 1 and κ2 = 0.

i=1, (cid:80)m
i=1 pi(cid:107)∇Fi(x)(cid:107)2 ≤ β2(cid:107) (cid:80)m

i=1 pi = 1, there exist
i=1 pi∇Fi(x)(cid:107)2+κ2.

5.2 ANALYSIS

We begin with showing two key lemmas. All the proofs can be found in Appendix.
Lemma 5.1. (Framework) Under Assumption 1 ∼ 3, if the learning rate satisﬁes η ≤ 1
ensures

2L , FedLAMA

1
K

K
(cid:88)

k=1

(cid:107)∇F (uk)(cid:107)2(cid:105)
(cid:104)

E

≤

2
ηK

E [F (u1) − F (u∗)] + 2ηLσ2

+

L2
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

.

5

m
(cid:88)

i=1

(pi)2

(5)

a) CIFAR-10 (ResNet20) b) CIFAR-100 (WRN28-10)05101520250.00.20.40.60.81.0normalized magnitude# of layers with increased interval delta lambda051015200.00.20.40.60.81.0normalized magnitude# of layers with increased interval delta lambdaUnder review as a conference paper at ICLR 2022

Lemma 5.2. (Model Discrepancy) Under Assumption 1 ∼ 4, if the learning rate satisﬁes η <
2(τmax−1)L , FedLAMA ensures

1

1
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

≤

2η2(τmax − 1)σ2
1 − A

+

Aκ2
L2(1 − A)

+

Aβ2
KL2(1 − A)

K
(cid:88)

k=1

(cid:104)

(cid:107)∇F (uk)(cid:107)2(cid:105)

,

E

(6)

where A = 4η2(τmax − 1)2L2 and τmax is the largest averaging interval across all the layers.

Based on Lemma 5.1 and 5.2, we analyze the convergence rate of FedLAMA as follows.
Theorem 5.3. Suppose all m local models are initialized to the same point u1. Under As-
if FedLAMA runs for K iterations and the learning rate satisﬁes η ≤
sumption 1 ∼ 4,

(cid:26)

min

1
2(τmax−1)L ,

√

L

1

2τmax(τmax−1)(2β2+1)

, FedLAMA ensures

(cid:27)

(cid:34)

E

1
K

K
(cid:88)

i=1

(cid:35)

(cid:107)∇F (uk)(cid:107)2

≤

4
ηK

(E [F (u1) − F (u∗)]) + 4η

m
(cid:88)

i=1

i Lσ2
p2

(7)

+ 3η2(τmax − 1)L2σ2 + 6η2τmax(τmax − 1)L2κ2,

where u∗ indicates a local minimum and τmax is the largest averaging interval across all the layers.

Remark 1. (Linear Speedup) With a sufﬁciently small diminishing learning rate and a large number
and
of training iterations, FedLAMA achieves linear speedup.
pi = 1

If the learning rate is η =

m , ∀i ∈ {1, · · · , m}, we have

√
m√
K

(cid:34)

E

1
K

K
(cid:88)

i=1

(cid:35)

(cid:107)∇F (uk)(cid:107)2

≤ O

(cid:19)

(cid:18) 1
√

mK

+ O

(cid:17)

(cid:16) m
K

(8)

If K > m3, the ﬁrst term on the right-hand side becomes dominant and it achieves linear speedup.

Remark 2. (Impact of Interval Increase Factor φ) The worst-case model discrepancy depends on the
largest averaging interval across all the layers, τmax = φτ (cid:48). The larger the interval increase factor φ,
the larger the model discrepancy terms in (7). In the meantime, as φ increases, the communication
frequency at the selected layers is proportionally reduced. So, φ should be appropriately tuned to
effectively reduce the communication cost while not much increasing the model discrepancy.

6 EXPERIMENTS

Experimental Settings – We evaluate FedLAMA using three representative benchmark datasets:
CIFAR-10 (ResNet20 (He et al. (2016))), CIFAR-100 (WideResNet28-10 (Zagoruyko & Komodakis
(2016))), and Federated Extended MNIST (CNN (Caldas et al. (2018))). We use TensorFlow 2.4.3
for local training and MPI for model aggregation. All our experiments are conducted on 4 compute
nodes each of which has 2 NVIDIA v100 GPUs.

Due to the limited compute resources, we simulate Federated Learning such that each process se-
quentially trains multiple models and then the models are aggregated across all the processes at
once. While it provides the same classiﬁcation results as the actual Federated Learning, the train-
ing time is serialized within each process. Thus, instead of wall-clock time, we consider the total
communication cost calculated as follows.

C =

L
(cid:88)

l=1

Cl =

L
(cid:88)

l=1

dim(ul) ∗ κl,

(9)

where κl is the total number of communications at layer l during the training.

6

Under review as a conference paper at ICLR 2022

Table 1:
batch size is 32 in all the experiments. The epoch budget is 300.

(IID data) CIFAR-10 classiﬁcation results. The number of workers is 128 and the local

LR
0.8
0.8
0.4
0.6
0.6

Base aggregation interval: τ (cid:48)
6
12
6
24
6

Interval increase factor: φ
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)
1 (FedAvg)
4 (FedLAMA)

Validation acc.
88.37 ± 0.02%
84.74 ± 0.05%
88.41 ±0.01%
80.34 ± 0.3%
86.21 ±0.1%

Comm. cost
100%
50%
62.33%
25%
42.17%

Table 2:
batch size is 32 in all the experiments. The epoch budget is 250.

(IID data) CIFAR-100 classiﬁcation results. The number of workers is 128 and the local

LR
0.6
0.6
0.5
0.6
0.5

Base aggregation interval: τ (cid:48)
6
12
6
24
6

Interval increase factor: φ
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)
1 (FedAvg)
4 (FedLAMA)

Validation acc.
76.50 ± 0.02%
66.97 ± 0.9%
76.02 ±0.01%
45.01 ± 1.1%
76.17 ±0.02%

Comm. cost
100%
50%
66.01%
25%
39.91%

Hyper-Parameter Settings – We use 128 clients in our experiments. The local batch size is set to 32
and the learning rate is tuned based on a grid search. For CIFAR-10 and CIFAR-100, we artiﬁcially
generate heterogeneous data distributions using Dirichlet’s distribution. When using Non-IID data,
we also consider partial device participation such that randomly chosen 25% of the clients participate
in training at every φτ (cid:48) iterations. We report the average accuracy across at least three separate runs.

6.1 CLASSIFICATION PERFORMANCE ANALYSIS

To evaluate the proposed model aggregation scheme, we keep all the other factors the same, such as
optimizer, the number of clients, the degree of heterogeneity, and compare the performance across
different model aggregation schemes. We compare the performance across three different model
aggregation settings as follows.

• Periodic full aggregation with an interval of τ (cid:48)
• Periodic full aggregation with an interval of φτ (cid:48)
• Layer-wise adaptive aggregation with intervals of τ (cid:48) and φ

The ﬁrst setting provides the baseline communication cost, and we compare it to the other settings’
communication costs. The third setting is FedLAMA with the base aggregation interval τ (cid:48) and the
interval increase factor φ. Due to the limited space, we present a part of experimental results that
deliver the key insights. More results can be found in Appendix.

Experimental Results with IID Data – We ﬁrst present CIFAR-10 and CIFAR-100 classiﬁcation
results under IID data settings. Table 1 and 2 show the CIFAR-10 and CIFAR-100 results, respec-
tively. Note that the learning rate is individually tuned for each setting using a grid search, and
we report the best settings. In both tables, the ﬁrst row shows the performance of FedAvg with
a short interval τ (cid:48) = 6. As the interval increases, FedAvg signiﬁcantly loses the accuracy while
the communication cost is proportionally reduced. FedLAMA achieves a comparable accuracy to
FedAvg with τ (cid:48) = 6 while its communication cost is similar to that of FedAvg with φτ (cid:48). These
results demonstrate that Algorithm 2 effectively ﬁnds the layer-wise interval settings that maximize
the communication cost reduction while minimizing the model discrepancy increase.

Experimental Results with Non-IID Data – We now evaluate the performance of FedLAMA using
non-IID data. FEMNIST is inherently heterogeneous such that it contains the hand-written digit
pictures collected from 3, 550 different writers. We use random 10% of the writers’ training samples
in our experiments. Table 3 shows the FEMNIST classiﬁcation results. The base interval τ (cid:48) is set
to 10. FedAvg (φ = 1) signiﬁcantly loses the accuracy as the aggregation interval increases. For
example, when the interval increases from 10 to 40, the accuracy is dropped by 2.1% ∼ 2.7%.
In contrast, FedLAMA maintains the accuracy when φ increases, while the communication cost
is remarkably reduced. This result demonstrates that FedLAMA effectively ﬁnds the best interval
setting that reduces the communication cost while maintaining the accuracy.

7

Under review as a conference paper at ICLR 2022

Table 3:
local batch size is 32 in all the experiments. The number of training iterations is 2, 000.

(Non-IID data) FEMNIST classiﬁcation results. The number of workers is 128 and the

LR

0.04

0.04

0.04

Base aggregation interval: τ (cid:48)
10
20
10
40
10
10
20
10
40
10
10
20
10
40
10

Interval increase factor: φ
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)
1 (FedAvg)
4 (FedLAMA)
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)
1 (FedAvg)
4 (FedLAMA)
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)
1 (FedAvg)
4 (FedLAMA)

active ratio

25%

50%

100%

Validation acc.
86.04 ± 0.01%
85.38 ± 0.02%
86.01 ±0.01%
83.97 ± 0.02%
85.61 ±0.02%
86.59 ± 0.01%
85.50 ± 0.02%
86.07 ±0.02%
83.92 ± 0.02%
85.77 ±0.02%
85.74 ± 0.03%
85.08 ± 0.01%
85.40 ±0.02%
83.62 ± 0.02%
84.67 ±0.02%

Comm. cost
100%
50%
52.83%
25%
29.97%
100%
50%
53.32%
25%
29.98%
100%
50%
51.86%
25%
29.98%

Table 4:
local batch size is 32 in all the experiments. The number of training iterations is 6, 000.

(Non-IID data) CIFAR-10 classiﬁcation results. The number of workers is 128 and the

LR

0.4

0.8

0.8

0.8

Base aggregation interval: τ (cid:48)
6
24
6
6
24
6
6
24
6
6
24
6

Interval increase factor: φ
1 (FedAvg)
1 (FedAvg)
4 (FedLAMA)
1 (FedAvg)
1 (FedAvg)
4 (FedLAMA)
1 (FedAvg)
1 (FedAvg)
4 (FedLAMA)
1 (FedAvg)
1 (FedAvg)
4 (FedLAMA)

active ratio

Dirichlet’s coeff.

25%

25%

100%

100%

0.1

0.5

0.1

0.5

Validation acc.
84.02 ± 0.1%
76.27 ± 0.08%
83.06 ±0.1%
87.59 ± 0.2%
83.36 ± 0.4%
86.57 ±0.02%
89.52 ± 0.05%
84.82 ± 0.06%
87.47 ±0.1%
90.53 ± 0.08%
85.68 ± 0.1%
87.45 ±0.05%

Comm. cost
100%
25%
39.52%
100%
25%
42.40%
100%
25%
42.49%
100%
25%
42.73%

Table 5:
local batch size is 32 in all the experiments. The number of training iterations is 6, 000.

(Non-IID data) CIFAR-100 classiﬁcation results. The number of workers is 128 and the

LR

0.4

0.4

0.4

0.4

Base aggregation interval: τ (cid:48)
6
12
6
6
12
6
6
12
6
6
12
6

Interval increase factor: φ
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)
1 (FedAvg)
1 (FedAvg)
2 (FedLAMA)

active ratio

Dirichlet’s coeff.

25%

25%

100%

100%

0.1

0.5

0.1

0.5

Validation acc.
79.15 ± 0.02%
76.16 ± 0.05%
78.63 ±0.03%
78.81 ± 0.1%
76.11 ± 0.05%
77.86 ±0.04%
79.77 ± 0.04%
77.71 ± 0.08%
79.07 ±0.1%
80.19 ± 0.05%
77.40 ± 0.06%
78.88 ±0.05%

Comm. cost
100%
50%
63.14%
100%
50%
63.20%
100%
50%
60.48%
100%
50%
61.73%

Table 4 and 5 show the non-IID CIFAR-10 and CIFAR-100 experimental results. We use Dirichlet’s
distribution to generate heterogeneous data across all the clients. The detailed settings regarding
Dirichlet’s distribution can be found in Appendix. The base aggregation interval τ (cid:48) is set to 6. The
interval increase factor φ is set to 2 for FedLAMA. Likely to the IID data experiments, we observe
that the periodic full averaging signiﬁcantly loses the accuracy as the model aggregation interval
increases, while it has a proportionally reduced communication cost. For both datasets, FedLAMA
achieves a comparable accuracy to the periodic full averaging with the interval of τ (cid:48) while having
the communication cost that is close to the periodic full averaging with the increased interval of
φτ (cid:48). Especially, FedLAMA works effectively even when the Dirichlet’s coefﬁcient is set to 0.1.
The coefﬁcient of 0.1 represents an extremely high degree of data heterogeneity in terms of not
only the number of samples per client but also the balance of the classes assigned to each client.
These results imply that FedLAMA is a practical algorithm for Federated Learning applications
with highly heterogeneous data distributions.

8

Under review as a conference paper at ICLR 2022

Figure 2: The number of communications at the individual layers. The communications are counted
during the whole training (non-IID data).

Figure 3: The total data size (communication cost) that correspond to Figure 2. The data size
comparison clearly shows where the performance gain of FedLAMA comes from.

6.2 COMMUNICATION EFFICIENCY ANALYSIS

We analyze the total number of communications and the accumulated data size to evaluate the com-
munication efﬁciency of FedLAMA. Figure 2 shows the total number of communications at the
individual layers. The τ (cid:48) is set to 6 and φ is 2 for FedLAMA. The key insight is that FedLAMA
increases the aggregation interval mostly at the output-side large layers. This means the dl value
shown in Equation (2) at the these layers are smaller than the others. Since these large layers take
up most of the total model parameters, the communication cost is remarkably reduced when their
aggregation intervals are increased. Figure 3 shows the layer-wise local data size shown in Equation
9. FedLAMA shows the signiﬁcantly smaller total data size than FedAvg. The extra computational
cost of FedLAMA is almost negligible since it calculates dl after each communication round only.
Therefore, given the virtually same computational cost, FedLAMA aggregates the local models at a
cheaper communication cost, and thus it improves the scalablity of Federated Learning.

We found that the amount of the reduced communication cost was not strongly affected by the
degree of data heterogeneity. As shown in Table 4 and 5, the reduced communication cost is similar
across different Dirichlet’s coefﬁcients and device participation ratios. That is, FedLAMA can be
considered as an effective model aggregation scheme regardless of the degree of data heterogeneity.

7 CONCLUSION

We proposed a layer-wise model aggregation scheme that adaptively adjusts the model aggregation
interval at run-time. Breaking the convention of aggregating the whole model parameters at once,
this novel model aggregation scheme introduces a ﬂexible communication strategy for scalable Fed-
erated Learning. Furthermore, we provide a solid convergence guarantee of FedLAMA under the
assumptions on the non-convex objective functions and the non-IID data distribution. Our empirical
study also demonstrates the efﬁcacy of FedLAMA for scalable and accurate Federated Learning.

9

13579111315171921230500100015001234050100150200135791113151719212325272905001000a) CIFAR-10 (ResNet20)b) CIFAR-100 (WRN28-10)c) FEMNIST (CNN)layer IDlayer IDlayer ID# of communicationsFedAvgFedLAMA13579111315171921232527290.00E+0001.00E+0092.00E+0093.00E+0094.00E+00913579111315171921230.00E+0002.00E+0074.00E+0076.00E+00712340.00E+0005.00E+0081.00E+0091.50E+009a) CIFAR-10 (ResNet20)b) CIFAR-100 (WRN28-10)c) FEMNIST (CNN)layer IDlayer IDlayer IDtotal data sizeFedAvgFedLAMAUnder review as a conference paper at ICLR 2022

Harmonizing FedLAMA with other advanced optimizers, gradient compression, and low-rank ap-
proximation methods is a promising future work.

10

Under review as a conference paper at ICLR 2022

8 CODE OF ETHICS

Our work does not deliver potentially harmful insights or conﬂicts of interests. We also do not ﬁnd
any potential inappropriate application or privacy/security issues. The datasets we used in our study
are all public benchmark datasets, and our source code will be opened once the paper is accepted.

9 REPRODUCIBILITY STATEMENT

The software versions, implementation details, hyper-parameter settings can be found in the ﬁrst
two paragraphs of Section 6. The entire source code used in our experiments will be published as
an open source once the paper is accepted. We believe one can exactly reproduce our experimental
results following the provided descriptions.

REFERENCES

Alyazeed Albasyoni, Mher Safaryan, Laurent Condat, and Peter Richt´arik. Optimal gradient com-

pression for distributed and federated learning. arXiv preprint arXiv:2010.03246, 2020.

Dan Alistarh, Demjan Grubic,

Qsgd:
Communication-efﬁcient sgd via gradient quantization and encoding. Advances in Neural In-
formation Processing Systems, 30:1709–1720, 2017.

Jerry Li, Ryota Tomioka, and Milan Vojnovic.

Dan Alistarh, Torsten Hoeﬂer, Mikael Johansson, Sarit Khirirat, Nikola Konstantinov, and C´edric
Renggli. The convergence of sparsiﬁed gradient methods. arXiv preprint arXiv:1809.10505,
2018.

Andrew Brock, Theodore Lim, James M Ritchie, and Nick Weston. Freezeout: Accelerate training

by progressively freezing layers. arXiv preprint arXiv:1706.04983, 2017.

Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Koneˇcn`y, H Brendan McMa-
han, Virginia Smith, and Ameet Talwalkar. Leaf: A benchmark for federated settings. arXiv
preprint arXiv:1812.01097, 2018.

Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik. Emnist: Extending mnist
to handwritten letters. In 2017 International Joint Conference on Neural Networks (IJCNN), pp.
2921–2926. IEEE, 2017.

Enmao Diao, Jie Ding, and Vahid Tarokh. Heteroﬂ: Computation and communication efﬁcient

federated learning for heterogeneous clients. arXiv preprint arXiv:2010.01264, 2020.

Kelam Goutam, S Balasubramanian, Darshan Gera, and R Raghunatha Sarma. Layerout: Freezing

layers in deep neural networks. SN Computer Science, 1(5):1–9, 2020.

Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, An-
drew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet
in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Sch¨olkopf. Measuring statistical de-
pendence with hilbert-schmidt norms. In International conference on algorithmic learning theory,
pp. 63–77. Springer, 2005.

Farzin Haddadpour, Mohammad Mahdi Kamani, Mehrdad Mahdavi, and Viveck R Cadambe. Lo-
cal sgd with periodic averaging: Tighter analysis and adaptive synchronization. arXiv preprint
arXiv:1910.13598, 2019.

Chaoyang He, Songze Li, Jinhyun So, Xiao Zeng, Mi Zhang, Hongyi Wang, Xiaoyang Wang, Pra-
neeth Vepakomma, Abhishek Singh, Hang Qiu, et al. Fedml: A research library and benchmark
for federated machine learning. arXiv preprint arXiv:2007.13518, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-
nition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp.
770–778, 2016.

11

Under review as a conference paper at ICLR 2022

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and
Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In
International Conference on Machine Learning, pp. 5132–5143. PMLR, 2020.

Jakub Koneˇcn`y, H Brendan McMahan, Felix X Yu, Peter Richt´arik, Ananda Theertha Suresh, and
Dave Bacon. Federated learning: Strategies for improving communication efﬁciency. arXiv
preprint arXiv:1610.05492, 2016.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural
network representations revisited. In International Conference on Machine Learning, pp. 3519–
3529. PMLR, 2019.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.

2009.

Adarsh Kumar, Arjun Balasubramanian, Shivaram Venkataraman, and Aditya Akella. Accelerat-
ing deep learning inference via freezing. In 11th {USENIX} Workshop on Hot Topics in Cloud
Computing (HotCloud 19), 2019.

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith.
Federated optimization in heterogeneous networks. arXiv preprint arXiv:1812.06127, 2018.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
Communication-efﬁcient learning of deep networks from decentralized data. In Artiﬁcial intelli-
gence and statistics, pp. 1273–1282. PMLR, 2017.

Ari S Morcos, Maithra Raghu, and Samy Bengio. Insights on representational similarity in neural

networks with canonical correlation. arXiv preprint arXiv:1806.05759, 2018.

Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. Svcca: Singular vector
canonical correlation analysis for deep learning dynamics and interpretability. arXiv preprint
arXiv:1706.05806, 2017.

Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani.
Fedpaq: A communication-efﬁcient federated learning method with periodic averaging and quan-
In International Conference on Artiﬁcial Intelligence and Statistics, pp. 2021–2031.
tization.
PMLR, 2020.

Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. Practical low-rank communication com-

pression in decentralized deep learning. In NeurIPS, 2020.

Hongyi Wang, Scott Sievert, Zachary Charles, Shengchao Liu, Stephen Wright, and Dimitris Pa-
pailiopoulos. Atomo: Communication-efﬁcient learning via atomic sparsiﬁcation. arXiv preprint
arXiv:1806.04090, 2018.

Hongyi Wang, Saurabh Agarwal, and Dimitris Papailiopoulos. Pufferﬁsh: Communication-efﬁcient

models at no extra cost. arXiv preprint arXiv:2103.03936, 2021.

Jianyu Wang and Gauri Joshi. Adaptive communication strategies to achieve the best error-runtime

trade-off in local-update sgd. arXiv preprint arXiv:1810.08313, 2018.

Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective in-
consistency problem in heterogeneous federated optimization. arXiv preprint arXiv:2007.07481,
2020.

Jianqiao Wangni, Jialei Wang, Ji Liu, and Tong Zhang. Gradient sparsiﬁcation for communication-

efﬁcient distributed optimization. arXiv preprint arXiv:1710.09854, 2017.

Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Tern-
grad: Ternary gradients to reduce communication in distributed deep learning. arXiv preprint
arXiv:1705.07878, 2017.

Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan
Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep
learning: Training bert in 76 minutes. arXiv preprint arXiv:1904.00962, 2019.

12

Under review as a conference paper at ICLR 2022

Sergey Zagoruyko and Nikos Komodakis.

Wide residual networks.

arXiv preprint

arXiv:1605.07146, 2016.

Minjia Zhang and Yuxiong He. Accelerating training of transformer-based language models with

progressive layer dropping. arXiv preprint arXiv:2010.13369, 2020.

13

Under review as a conference paper at ICLR 2022

A APPENDIX

A.1 CONVERGENCE ANALYSIS

Herein, we provide the proofs of the lemmas and theorem shown in Section 5.

A.1.1 PRELIMINARIES

FedLAMA periodically chooses a few layers that will be less frequently synchronized. We call these
layers Least Critical Layers (LCL) for short.
Notations – All vectors in this paper are column vectors. x ∈ Rd denotes the parameters of one
local model and m is the number of workers. The stochastic gradient computed from a single
training data point ξ is denoted by g(x, ξ). For convenience, we use g(x) instead. The full batch
gradient is denoted by ∇F (x). We use (cid:107) · (cid:107) and (cid:107) · (cid:107)op to denote l2 norm and matrix operator norm,
respectively.

Objective Function – In this paper, we consider federated optimization problems as follows.

(cid:34)

min
x∈Rd

F (x) :=

(cid:35)

piFi(x)

,

m
(cid:88)

i=1

(10)

where pi = ni/n is the ratio of local data to the total dataset, and Fi(x) = 1
ξ∈D fi(x, ξ) is the
ni
local objective function of client i. n is the global dataset size and ni is the local dataset size. Note
that, by deﬁnition, (cid:80)m
Averaging Matrix – We deﬁne a time-varying averaging matrix Wk ∈ Rdm×dm as follows.

i=1 pi = 1.

(cid:80)

Wk =






P,
J,
I,

if k mod τmin is 0
if k mod τmax is 0
otherwise

(11)

I is an identity matrix, P is also a time-varying averaging matrix, and J is a full averaging matrix.
First, P1
i is a d × d diagonal matrix that has 1 for the diagonal elements that correspond to the LCL
parameters and pi for all the other diagonal elements. Likewise, P0
i is another d × d diagonal matrix
that has 0 for the diagonal elements that correspond to the LCL parameters and pi for all the other
diagonal elements. Then, P is deﬁned as follows.

P =

(cid:26)P1, for m diagonal blocks
P0, for all the other blocks

(12)

The ith block column of P consists of P1

i and P0

i following the above deﬁnition.

Here we present an example of P where m = 2 and d = 2. In this example, p0 = 1/3 and p1 = 2/3.
Saying the LCL is the second parameter, P is deﬁned as follows.

P1

0 =

(cid:20) 1
3
0

(cid:21)

0
1

, P0

0 =

(cid:20) 1
3
0

(cid:21)

0
0

, P1

1 =

(cid:20) 2
3
0

(cid:21)
0
1

, P0

1 =

(cid:20) 2
3
0

(cid:21)
0
0

P =

(cid:21)

(cid:20)P1
P0

0 P0
1
0 P1
1

=






1
3
0
1
3
0

0
1
0
0

2
3
0
2
3
0


0
0

 .
0
1

(13)

(14)

The full-averaging matrix J is deﬁned as follows. First, Ji is a d × d diagonal matrix that has pi for
the diagonal elements. Then, J consists of m × m blocks of Ji such that each column block is m of
Ji blocks. Here we present an example of J where m = 2 and d = 2 as follows.

J0 =

(cid:20) 1
3
0

(cid:21)

0
1
3

, J1 =

(cid:20) 2
3
0

(cid:21)

0
2
3

14

(15)

Under review as a conference paper at ICLR 2022

J =

(cid:21)

(cid:20)J0 J1
J0 J1

=







1
3
0
1
3
0

0
1
3
0
1
3

2
3
0
2
3
0







0
2
3
0
2
3

.

(16)

The averaging matrix P and J have the following properties:

1. P1dm = 1dm, J1dm = 1dm.
2. The product of any two averaging matrices consists only of diagonal block matrices because

all the blocks in P and J are diagonal.

3. PJ = JP = J regardless of which layers are chosen as the LCL.
4. PP = P regardless of which layers are chosen as the LCL.

Vectorization – We deﬁne a vectorized form of m local model parameters xk ∈ Rdm, its stochastic
gradients gk ∈ Rdm, and the full gradients fk ∈ Rdm as follows
xk = vec (cid:8)x1
k, x2
gk = vec (cid:8)g1(x1
fk = vec (cid:8)∇F1(x1

k), ∇F2(x2
The full model aggregation can be written using the vectorized form of local models xk and the
averaging matrix J as follows.

k, · · · , xm
k
k), · · · , gm(xm
k), g2(x2

k), · · · , ∇Fm(xm

k )(cid:9) .

k )(cid:9)

(17)

(cid:9)

Jxk =







1
3
0
1
3
0

0
1
3
0
1
3

2
3
0
2
3
0







0
2
3
0
2
3








x(1,1)
k
x(1,2)
k
x(2,1)
k
x(2,2)
k








=








k

(x(1,1)
(x(1,2)
(x(1,1)
(x(1,2)

k + 2x(2,1)
k + 2x(2,2)
k + 2x(2,1)
k + 2x(2,2)

k

k

k








)/3
)/3
)/3
)/3

(18)

where x(i,j)

k

is the jth model parameter of local model i at iteration k.

We also deﬁne the following additional vectorized forms of the weighted model parameters and
gradients for convenience.

ˆxk = vec (cid:8)√
ˆgk = vec (cid:8)√
ˆfk = vec (cid:8)√

√

p2x2
p1x1
k,
√
p1g1(x1
k),
p1∇F1(x1

k),

k, · · · ,
p2g2(x2
√

√

pmxm
k
k), · · · ,

(cid:9)
√

p2∇F2(x2

k), · · · ,

pmgm(xm
√

k )(cid:9)
pm∇Fm(xm

k )(cid:9)

(19)

Assumptions – We analyze the convergence rate of FedLAMA under the following assumptions.

1. (Smoothness). Each local objective function is L-smooth, that is, (cid:107)∇Fi(x) − ∇Fi(y)(cid:107) ≤

L(cid:107)x − y(cid:107), ∀i ∈ {1, · · · , m}.

2. (Unbiased Gradient). The stochastic gradient at each client is an unbiased estimator of the

local full-batch gradient: Eξ[gi(x, ξ)] = ∇Fi(x).

3. (Bounded Variance). The stochastic gradient at each client has bounded variance:

Eξ[(cid:107)gi(x, ξ) − ∇Fi(x)(cid:107)2 ≤ σ2], ∀i ∈ {1, · · · , m}, σ2 ≥ 0.
4. (Bounded Dissimilarity). For any sets of weights {pi ≥ 0}m

constants β2 ≥ 1 and κ2 ≥ 0 such that (cid:80)m
If local objective functions are identical to each other, β2 = 1 and κ2 = 0.

i=1, (cid:80)m
i=1 pi(cid:107)∇Fi(x)(cid:107)2 ≤ β2(cid:107) (cid:80)m

i=1 pi = 1, there exist
i=1 pi∇Fi(x)(cid:107)2+κ2.

15

Under review as a conference paper at ICLR 2022

A.1.2 PROOFS

Theorem 5.1. Suppose all m local models are initialized to the same point u1. Under As-
if FedLAMA runs for K iterations and the learning rate satisﬁes η ≤
sumption 1 ∼ 4,

(cid:26)

min

1
2(τmax−1)L ,

√

L

1

2τmax(τmax−1)(2β2+1)

, FedLAMA ensures

(cid:27)

(cid:34)

E

1
K

K
(cid:88)

i=1

(cid:35)

(cid:107)∇F (uk)(cid:107)2

≤

4
ηK

(E [F (u1) − F (u∗)]) + 4η

m
(cid:88)

i=1

i Lσ2
p2

+ 3η2(τmax − 1)L2σ2 + 6η2τmax(τmax − 1)L2κ2,

(20)

where u∗ indicates a local minimum.

Proof. Based on Lemma 5.1 and 5.2, we have

1
K

K
(cid:88)

k=1

(cid:107)∇F (uk)(cid:107)2(cid:105)
(cid:104)

E

≤

2
ηK

(E [F (u1) − F (u∗)]) + 2η

m
(cid:88)

i=1

i Lσ2
p2

(cid:32)

+ L2

η2(τmax − 1)σ2
j
1 − A

+

Aβ2
KL2(1 − A)

K
(cid:88)

k=1

(cid:104)

(cid:107)∇F (uk)(cid:107)2(cid:105)

E

+

Aκ2
L2(1 − A)

(cid:33)

.

After re-writing the left-hand side and a minor rearrangement, we have

1
K

K
(cid:88)

k=1

(cid:104)

(cid:107)∇F (uk)(cid:107)2(cid:105)

E

≤

2
ηK

(E [F (u1) − F (u∗)]) + 2η

m
(cid:88)

i=1

i Lσ2
p2

(cid:107)∇F (uk)(cid:107)2(cid:105)

+

1
K

+ L2

(cid:104)

K
(cid:88)

E

k=1

Aβ2
1 − A
(cid:18) η2(τmax − 1)σ2
1 − A

+

Aκ2
L2(1 − A)

(cid:19)

.

By moving the third term on the right-hand side to the left-hand side, we have

1
K

K
(cid:88)

(cid:18)

1 −

k=1

(cid:19)

Aβ2
1 − A

(cid:107)∇jF (uk)(cid:107)2(cid:105)
(cid:104)

E

≤

2
ηK

(E [F (u1) − F (u∗)]) + 2η

m
(cid:88)

i Lσ2
p2

+ L2

(cid:18) η2(τmax − 1)σ2
1 − A

+

i=1
Aκ2
L2(1 − A)

(cid:19)

.

(21)

If A ≤ 1

2β2+1 , then Aβ2

1−A ≤ 1

2 . Therefore, (21) can be simpliﬁed as follows.

1
K

K
(cid:88)

k=1

(cid:104)

(cid:107)∇F (uk)(cid:107)2(cid:105)

E

≤

4
ηK

(E [F (u1) − F (u∗)]) + 4η

m
(cid:88)

i=1

i Lσ2
p2

(22)

+ 2L2

(cid:18) η2(τmax − 1)σ2
1 − A

(cid:19)

+ 2

Aκ2
1 − A

.

The learning rate condition A ≤ 1
2β2 ≤ 2
1

3 , and thus

1−A ≤ 2

1

3 . Therefore, we have

2β2+1 also ensures that

1

1−A ≤ 1 + 1

2β2 . Based on Assumption 4,

1
K

K
(cid:88)

k=1

(cid:104)

(cid:107)∇F (uk)(cid:107)2(cid:105)

E

≤

4
ηK

(E [F (u1) − F (u∗)]) + 4η

m
(cid:88)

i=1

p2
i Lσ2

+ 3η2(τmax − 1)L2σ2 + 6η2τmax(τmax − 1)L2κ2.

We complete the proof.

16

Under review as a conference paper at ICLR 2022

Learning Rate Constraints – In Theorem 5.3, we have two learning rate constraints, one from (22)
and the other from (51) as follows.

A <

1
2β2 + 1

from (22)

A < 1

from (51)

After a minor rearrangement, we have a uniﬁed learning rate constraint as follows.

(cid:40)

η ≤ min

1
2(τmax − 1)L

,

1
L(cid:112)2τmax(τmax − 1)(2β2 + 1)

(cid:41)

Lemma 5.1. (Framework) Under Assumption 1 ∼ 3, if the learning rate satisﬁes η ≤ 1
ensures

2L , FedLAMA

1
K

K
(cid:88)

k=1

(cid:107)∇F (uk)(cid:107)2(cid:105)
(cid:104)

E

≤

2
ηK

E [F (u1) − F (u∗)] + 2ηLσ2

+

L2
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

.

m
(cid:88)

i=1

(pi)2

(23)

Proof. Based on Assumption 1, we have

(cid:34)

E [F (uk+1) − F (uk)] ≤ −ηE

(cid:104)∇F (uk),

m
(cid:88)

(cid:35)

pigi(xi

k)(cid:105)

+

(cid:124)

i=1
(cid:123)(cid:122)
T1

(cid:125)

(cid:34)

First, T1 can be rewritten as follows.
(cid:0)gi(xi

(cid:104)∇F (uk),

T1 = E

m
(cid:88)

pi

(cid:34)

k) − ∇Fi(xi

+ E

(cid:104)∇F (uk),

(cid:35)
k)(cid:1)(cid:105)

(cid:35)

pi∇Fi(xi

k)(cid:105)

m
(cid:88)

i=1

η2L
2

E

(cid:124)

pigi(xi
k)

m
(cid:88)





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

i=1

(cid:123)(cid:122)
T2

2

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(24)



(cid:125)

(cid:34)

= E

(cid:104)∇F (uk),

i=1
m
(cid:88)

i=1

pi∇Fi(xi

(cid:35)
k)(cid:105)

=

1
2

(cid:107)∇F (uk)(cid:107)2 +

1
2

E





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

m
(cid:88)

i=1

pi∇Fi(xi
k)

2
 −

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

1
2

E





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

∇F (uk) −

(cid:13)
(cid:13)
pi∇Fi(xi
(cid:13)
k)
(cid:13)
(cid:13)

2
 ,

m
(cid:88)

i=1

(25)

where the last equality holds based on a basic equality: 2a(cid:62)b = (cid:107)a(cid:107)2 + (cid:107)b(cid:107)2 − (cid:107)a − b(cid:107)2 .

Then, T2 can be bounded as follows.

T2 = E

= E

≤ 2 E





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)




m
(cid:88)

i=1

m
(cid:88)

i=1

pi

pi

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

m
(cid:88)

i=1

(cid:0)gi(xi

k) − E (cid:2)gi(xi

k)(cid:3)(cid:1) +

m
(cid:88)

i=1

pi E (cid:2)gi(xi

(cid:0)gi(xi

k) − ∇Fi(xi

k)(cid:1) +

m
(cid:88)

pi∇Fi(xi
k)

(cid:0)gi(xi

pi

k) − ∇Fi(xi

k)(cid:1)

i=1
2
 + 2 E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

m
(cid:88)

i=1

2



(cid:13)
(cid:13)
k)(cid:3)
(cid:13)
(cid:13)
(cid:13)
2



(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

m
(cid:88)

i=1

= 2

m
(cid:88)

i=1

p2
i E

(cid:104)(cid:13)
(cid:13)gi(xi

k) − ∇Fi(xi

k)(cid:13)
(cid:13)

2(cid:105)

+ 2 E

≤ 2σ2

m
(cid:88)

i=1

p2
i + 2 E





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

m
(cid:88)

i=1

(cid:13)
(cid:13)
pi∇Fi(xi
(cid:13)
k)
(cid:13)
(cid:13)

2
 ,

17

2



(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
2

pi∇Fi(xi
k)

(cid:13)
(cid:13)
pi∇Fi(xi
(cid:13)
k)
(cid:13)
(cid:13)



(26)

Under review as a conference paper at ICLR 2022

where the last equality holds because gi(xi
and the last inequality follows Assumption 3.

k) − ∇Fi(xi

k) has 0 mean and is independent across i,

By plugging in (25) and (26) into (24), we have the following.

E [F (uk+1) − F (uk)] ≤ −

+

η
2

η
2

(cid:107)∇F (uk)(cid:107)2 −

η
2

E





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

m
(cid:88)

i=1

pi∇Fi(xi
k)

2



(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

∇F (uk) −

(cid:13)
(cid:13)
pi∇Fi(xi
(cid:13)
k)
(cid:13)
(cid:13)

2
 + η2Lσ2

m
(cid:88)

i=1

m
(cid:88)

i=1

p2
i





E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)




+ η2L E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

m
(cid:88)

i=1

(cid:13)
(cid:13)
pi∇Fi(xi
(cid:13)
k)
(cid:13)
(cid:13)

2



= −

+

η
2

η
2

(cid:107)∇F (uk)(cid:107)2 −

η
2

(1 − 2ηL) E





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

∇F (uk) −

m
(cid:88)

i=1

E

(cid:13)
(cid:13)
pi∇Fi(xi
(cid:13)
k)
(cid:13)
(cid:13)

m
(cid:88)





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:13)
(cid:13)
pi∇Fi(xi
(cid:13)
k)
(cid:13)
(cid:13)

2



i=1
2
 + η2Lσ2

m
(cid:88)

i=1

p2
i

If η ≤ 1

2L , it follows

E [F (uk+1) − F (uk)]
η

≤ −

+

≤ −

+

≤ −

1
2

1
2

1
2

1
2

1
2

(cid:107)∇F (uk)(cid:107)2 + ηLσ2

m
(cid:88)

i=1

p2
i





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

∇F (uk) −

E

m
(cid:88)

i=1

pi∇Fi(xi
k)

2



(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:107)∇F (uk)(cid:107)2 + ηLσ2

m
(cid:88)

i=1

p2
i

(27)

m
(cid:88)

i=1

pi E

(cid:104)(cid:13)
(cid:13)∇Fi(uk) − ∇Fi(xi

k)(cid:13)
(cid:13)

2(cid:105)

(cid:107)∇F (uk)(cid:107)2 + ηLσ2

m
(cid:88)

i=1

p2
i +

L2
2

m
(cid:88)

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

,

where (27) holds based on the convexity of (cid:96)2 norm and Jensen’s inequality.

By taking expectation and averaging across K iterations, we have.

1
K

K
(cid:88)

k=1

E [F (uk+1) − F (uk)]
η

≤ −

1
2K

K
(cid:88)

k=1

(cid:107)∇F (uk)(cid:107)2 + ηLσ2

m
(cid:88)

i=1

p2
i

+

L2
2K

K
(cid:88)

m
(cid:88)

k−1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

.

18

Under review as a conference paper at ICLR 2022

After a minor rearrangement, we have a telescoping sum as follows.

1
K

K
(cid:88)

k=1

(cid:104)

(cid:107)∇F (uk)(cid:107)2(cid:105)

E

≤

2
ηK

E [F (u1) − F (uk+1)] + 2ηLσ2

m
(cid:88)

i=1

p2
i

+

L2
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

≤

2
ηK

E [F (u1) − F (u∗)] + 2ηLσ2

+

L2
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

,

m
(cid:88)

i=1

p2
i

where u∗ indicates the local minimum. Here, we complete the proof.

Lemma 5.2. (Model Discrepancy) Under Assumption 1 ∼ 4, if the learning rate satisﬁes η <
2(τmax−1)L , FedLAMA ensures

1

1
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

≤

2η2(τmax − 1)σ2
1 − A

+

Aκ2
L2(1 − A)

+

Aβ2
KL2(1 − A)

K
(cid:88)

k=1

(cid:104)

(cid:107)∇F (uk)(cid:107)2(cid:105)

,

E

(28)

where A = 4η2(τmax − 1)2L2 and τmax is the largest averaging interval across all the layers.

Proof. We begin with rewriting the weighted average of the squared distance using the vectorized
form of the local models as follows.

m
(cid:88)

i=1

(cid:13)
(cid:13)uk − xi
k

(cid:13)
2
(cid:13)

pi

=

√

(cid:13)
(cid:13)

(cid:0)uk − xi

k

(cid:1)(cid:13)
2
(cid:13)

pi

m
(cid:88)

i=1

= (cid:107)Jˆxk − ˆxk(cid:107)2
= (cid:107)(J − I)ˆxk(cid:107)2 ,

(29)

where (29) holds by the commutative property of multiplication.

Then, according to the parameter update rule, we have

(J − I)ˆxk = (J − I)Wk−1(ˆxk−1 − ηˆgk−1)

= (J − I)Wk−1ˆxk−1 − (J − Wk−1)ηˆgk−1,

(30)

where (30) holds because JW = J based on the averaging matrix property 3, and IW = W.

Then, expanding the expression of xk−1, we have

(J − I)ˆxk = (J − I)Wk−1(Wk−2(ˆxk−2 − ηˆgk−2)) − (J − Wk−1)ηˆgk−1

= (J − I)Wk−1Wk−2ˆxk−2 − (J − Wk−1Wk−2)ηˆgk−2 − (J − Wk−1)ηˆgk−1.

Repeating the same procedure for ˆxk−2, ˆxk−3, · · · , ˆx2, we have

(J − I)ˆxk = (J − I)

k−1
(cid:89)

s=1

Wsˆx1 − η

k−1
(cid:88)

s=1

(J −

k−1
(cid:89)

l=s

Wl)ˆgs

k−1
(cid:88)

(J −

= −η

s=1

k−1
(cid:89)

l=s

Wl)ˆgs,

(31)

where (31) holds because xi

1 is the same across all the workers and thus (J − I)ˆx1 = 0.

19

Under review as a conference paper at ICLR 2022

Based on (31), we have

1
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

=

=

=

1
K

1
K

1
K

K
(cid:88)

(cid:16)

(cid:107)(J − I)ˆxk(cid:107)2(cid:105)(cid:17)
(cid:104)

E


η2 E


η2 E

k=1

K
(cid:88)

k=1

K
(cid:88)

k=1










(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

k−1
(cid:88)

s=1

k−1
(cid:88)

s=1

k−1
(cid:89)

l=s

k−1
(cid:89)

l=s

(J −

(J −

Wl)ˆgs





2



(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

Wl)(ˆgs − ˆfs) +

k−1
(cid:88)

(J −

s=1

k−1
(cid:89)

l=s

Wl)ˆfs

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

2









≤

2η2
K

E








K
(cid:88)

k=1
(cid:124)





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

k−1
(cid:88)

s=1

(J −

k−1
(cid:89)

l=s
(cid:123)(cid:122)
T3

(cid:13)
(cid:13)
Wl)(ˆgs − ˆfs)
(cid:13)
(cid:13)
(cid:13)

2

E

K
(cid:88)

k=1
(cid:124)

+



(cid:125)





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

k−1
(cid:88)

(J −

s=1

(cid:123)(cid:122)
T4

k−1
(cid:89)

l=s

Wl)ˆfs

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

2









(cid:125)

(32)

where (32) holds based on the convexity of (cid:96)2 norm and Jensen’s inequality. Now, we focus on
bounding T3 and T4, separately.

Bounding T3

(cid:13)
(cid:13)
Wl)(ˆgs − ˆfs)
(cid:13)
(cid:13)
(cid:13)

2



K
(cid:88)

k=1

E





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

k−1
(cid:88)

s=1

(J −

K
(cid:88)

k−1
(cid:88)

E

=

k−1
(cid:89)

l=s


(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)



(J −

(cid:13)
(cid:13)
Wl)(ˆgs − ˆfs)
(cid:13)
(cid:13)
(cid:13)

2



k−1
(cid:89)

l=s

k=1

s=1

K
(cid:88)

k−1
(cid:88)

k=1

s=1

≤


(cid:13)
(cid:13)
2
(cid:13)(ˆgs − ˆfs)
(cid:13)
(cid:13)

(cid:13)

E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(J −

k−1
(cid:89)

l=s

Wl)

2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

op



 ,

(33)

(34)

where (33) holds because ˆgs − ˆfs has 0 mean and independent across s, and (34) holds based on
Lemma A.1.

20

Under review as a conference paper at ICLR 2022

Without loss of generality, we replace k with aτmax + b, where a is the communication round index
and b is the iteration index within each communication round. Then, we have

K/τmax−1
(cid:88)

τmax(cid:88)

aτmax+b−1
(cid:88)

a=0

b=1

s=1


(cid:13)
(cid:13)
2
(cid:13)(ˆgs − ˆfs)
(cid:13)
(cid:13)

(cid:13)

E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(J −

k−1
(cid:89)

l=s





Wl)

2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

op

K/τmax−1
(cid:88)

τmax(cid:88)

aτ
(cid:88)

a=0

b=1

s=1


(cid:13)
(cid:13)
2
(cid:13)(ˆgs − ˆfs)
(cid:13)
(cid:13)

(cid:13)

E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(J −

aτmax+b−1
(cid:89)

Wl)





2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

op

K/τmax−1
(cid:88)

τmax(cid:88)

aτmax+b−1
(cid:88)

E

+


(cid:13)
(cid:13)(ˆgs − ˆfs)
(cid:13)


l=s

(cid:13)
2
(cid:13)
(cid:13)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(J −

aτmax+b−1
(cid:89)

l=s

Wl)





2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)


op

a=0

b=1

K/τmax−1
(cid:88)

τmax(cid:88)

a=0

b=1

aτmax+b−1
(cid:88)

s=aτmax+1

(cid:13)
(cid:13)
2
(cid:13)(ˆgs − ˆfs)
(cid:13)
(cid:13)

(cid:13)

E

s=aτmax+1

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(J −

aτmax+b−1
(cid:89)

l=s

Wl)

2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

op

K/τmax−1
(cid:88)

τmax(cid:88)

aτ +b−1
(cid:88)

a=0

b=1

s=aτmax+1

(cid:20)(cid:13)
(cid:13)
(cid:13)(ˆgs − ˆfs)
(cid:13)
(cid:13)
(cid:13)

2(cid:21)

E

K/τmax−1
(cid:88)

τmax(cid:88)

aτmax+b−1
(cid:88)

m
(cid:88)

pi E

(cid:104)(cid:13)
(cid:13)(gi(xi

s) − ∇Fi(xi

s))(cid:13)
(cid:13)

2(cid:105)

a=0

b=1

s=aτmax+1

i=1

K/τmax−1
(cid:88)

τmax(cid:88)

aτmax+b−1
(cid:88)

m
(cid:88)

piσ2

a=0

b=1

s=aτmax+1

i=1

K/τmax−1
(cid:88)

τmax(cid:88)

(b − 1)σ2 =

K/τmax−1
(cid:88)

a=0

τmax(τmax − 1)
2

σ2

a=0
b=1
(τmax − 1)
2

σ2.

≤ K



(35)

(36)

(37)

(38)

=

=

=

=

≤

=

Remember FedLAMA synchronizes the whole parameters at least once after every τmax iterations.
Thus, (35) holds because (cid:81)aτmax+b−1
Wl
becomes 0. (36) holds based on Lemma A.2. (37) holds based on Assumption 3.

Wl is J when s ≤ aτmax, and thus J − (cid:81)aτmax+b−1

l=s

l=s

Bounding T4

21

Under review as a conference paper at ICLR 2022

K−τmax(cid:88)

E

k=1





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

k−1
(cid:88)

s=1

(J −

k−1
(cid:89)

l=s

Wl)ˆfs

2



(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)









(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

E

E

K/τmax−1
(cid:88)

τmax(cid:88)

a=0

b=1

K/τmax−1
(cid:88)

τmax(cid:88)

a=0

b=1

K/τmax−1
(cid:88)

τmax(cid:88)

aτ +b−1
(cid:88)

aτmax+b−1
(cid:89)

(J −

Wl)ˆfs

s=1

l=s

aτmax+b−1
(cid:88)

(J −

aτmax+b−1
(cid:89)

Pl)ˆfs

2

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)



2



(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

s=aτmax+1



(b − 1)

aτmax+b−1
(cid:88)

a=0

b=1

s=aτmax+1

K/τmax−1
(cid:88)

τmax(cid:88)

a=0

b=1



(b − 1)

K/τmax−1
(cid:88)

τmax(cid:88)

(cid:32)

(b − 1)

aτmax+b−1
(cid:88)

s=aτmax+1

aτmax+b−1
(cid:88)

a=0

b=1

s=aτmax+1

l=s





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(J −





(cid:13)
ˆfs
(cid:13)
(cid:13)

E

E

(cid:20)(cid:13)
ˆfs
(cid:13)
(cid:13)

E

(cid:13)
2
(cid:13)
(cid:13)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
2(cid:21)(cid:33)
(cid:13)
(cid:13)
(cid:13)

aτmax+b−1
(cid:89)

Pl)ˆfs

l=s

2







(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(J −

aτmax+b−1
(cid:89)

l=s

Pl)

2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

op

=

=

≤

≤

≤

(39)

(40)

(41)

(42)









≤

τmax(τmax − 1)
2

K/τmax−1
(cid:88)

(cid:32)aτmax+τmax−1
(cid:88)

a=0

s=aτmax+1

(cid:20)(cid:13)
ˆfs
(cid:13)
(cid:13)

2(cid:21)(cid:33)
(cid:13)
(cid:13)
(cid:13)

E

≤

=

τmax(τmax − 1)
2

(cid:20)(cid:13)
ˆfk
(cid:13)
(cid:13)

E

2(cid:21)

(cid:13)
(cid:13)
(cid:13)

K
(cid:88)

k=1

τmax(τmax − 1)
2

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)∇Fi(xi

k)(cid:13)
(cid:13)

2(cid:105)

,

(43)

where (39) holds because J − (cid:81)aτmax+b−1
Pl becomes 0 when s ≤ aτ . (40) holds based on the
convexity of (cid:96)2 norm and Jensen’s inequality. (41) holds based on Lemma A.1. (42) holds based on
Lemma A.2.

l=s

Final Result

By plugging in (38) and (43) into (32), we have

1
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

(cid:32)

K

≤

2η2
K

(τmax − 1)
2

σ2 +

τmax(τmax − 1)
2

(cid:32) K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:33)(cid:33)

(cid:104)(cid:13)
(cid:13)∇Fi(xi

k)(cid:13)
(cid:13)

2(cid:105)

= η2(τmax − 1)σ2 +

η2τmax(τmax − 1)
K

(cid:32) K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)∇Fi(xi

k)(cid:13)
(cid:13)

2(cid:105)

(cid:33)

(44)

The local gradient term on the right-hand side in (44) can be rewritten using the following inequality.

(cid:104)(cid:13)
(cid:13)∇Fi(xi

k)(cid:13)
(cid:13)

2(cid:105)

E

= E

2(cid:105)

(cid:104)(cid:13)
(cid:13)∇Fi(xi
(cid:104)(cid:13)
(cid:13)∇Fi(xi
(cid:104)(cid:13)
(cid:13)uk − xi
k

k) − ∇Fi(uk) + ∇Fi(uk)(cid:13)
(cid:13)
(cid:107)∇Fi(uk)(cid:107)2(cid:105)
(cid:104)
k) − ∇Fi(uk)(cid:13)
(cid:13)
(cid:104)
2(cid:105)

+ 2 E
(cid:107)∇Fi(uk)(cid:107)2(cid:105)

+ 2 E

2(cid:105)

(cid:13)
(cid:13)

,

≤ 2 E

≤ 2L2 E

(45)

(46)

22

Under review as a conference paper at ICLR 2022

where (45) holds based on the convexity of (cid:96)2 norm and Jensen’s inequality.

Plugging in (46) into (44), we have

1
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

≤ η2(τmax − 1)σ2 +

2η2τmax(τmax − 1)L2
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

+

2η2τmax(τmax − 1)
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)

(cid:107)∇Fi(uk)(cid:107)2(cid:105)

(47)

After a minor rearranging, we have

1
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

≤

η2(τmax − 1)σ2
1 − 2η2τmax(τmax − 1)L2

+

2η2τmax(τmax − 1)
K(1 − 2η2τmax(τmax − 1)L2)

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)

(cid:107)∇Fi(uk)(cid:107)2(cid:105)

(48)

Let us deﬁne A = 2η2τmax(τmax − 1)L2. Then (48) is simpliﬁed as follows.

1
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

≤

η2(τmax − 1)σ2
1 − A

+

A
KL2(1 − A)

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:107)∇Fi(uk)(cid:107)2(cid:105)
(cid:104)

Based on Assumption 4, we have

1
K

K
(cid:88)

m
(cid:88)

k=1

i=1

pi E

(cid:104)(cid:13)
(cid:13)uk − xi
k

2(cid:105)

(cid:13)
(cid:13)

≤

η2(τmax − 1)σ2
1 − A

+

Aβ2
KL2(1 − A)

=

η2(τmax − 1)σ2
1 − A

+

Aβ2
KL2(1 − A)

K
(cid:88)

k=1

K
(cid:88)

k=1





(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

m
(cid:88)

i=1

pi∇Fi(uk)

2
 +

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

Aκ2
L2(1 − A)

E

(cid:107)∇F (uk)(cid:107)2(cid:105)
(cid:104)

E

+

Aκ2
L2(1 − A)

,

(49)

(50)

where (50) holds based on the deﬁnition of the objective function (10).

Note that (49) is true only when 1 − A > 0. Thus, after a minor rearrangement, we have a learning
rate constraint as follows.

η <

1
2(τmax − 1)L

(51)

Here, we complete the proof.

A.1.3 PROOF OF OTHER LEMMAS

Lemma A.1. Consider a real matrix A ∈ Rmdj ×mdj and a real vector b ∈ Rmdj . If b (cid:54)= 0mdj ,
we have

(cid:107)Ab(cid:107) ≤ (cid:107)A(cid:107)op(cid:107)b(cid:107)

(52)

23

Under review as a conference paper at ICLR 2022

Proof.

(cid:107)Ab(cid:107)2 =

(cid:107)Ab(cid:107)2
(cid:107)b(cid:107)2 (cid:107)b(cid:107)2
op(cid:107)b(cid:107)2

≤ (cid:107)A(cid:107)2

where (53) holds based on the deﬁnition of operator norm.

Lemma A.2. Suppose an md × md averaging matrix P and the full-averaging matrix J, then

regardless of which layers are chosen as the LCL.

(cid:107)J − P(cid:107)2

op = 1.

(53)

(54)

Proof. First, by the deﬁnition of averaging matrix P, all the columns that do not correspond to the
LCL are zeroed out in J − P. Then, based on the averaging matrix property 1 and 2, the remaining
columns in P has 1 at all different rows. By the deﬁnition of J, all the non-zero elements in ith
column are the same pi, i ∈ {1, · · · , m}. Consequently, the remaining columns in J − P are
always orthogonal regardless of which layers are chosen as the LCL, and thus the eigenvalues of
J − P are either 1 or −1. Finally, by the deﬁnition of the matrix operator norm, (cid:107)J − P(cid:107)2
op =
max{|λ(J − P)|} = 1, where λ(·) indicates the eigenvalues of the input matrix.

24

Under review as a conference paper at ICLR 2022

Figure 4: The learning curves of CIFAR-10 (ResNet20) training (128 clients). a): The curves for
IID data distribution. b): The curves for non-IID data distribution (α = 0.1). FedAvg (x) indicates
FedAvg with the interval of x. FedLAMA (x, y) indicates FedLAMA with the base interval of x
and the interval increase factor of y. As the aggregation interval increases, FedAvg rapidly loses the
convergence speed, and it results in achieving a lower validation accuracy within the ﬁxed iteration
budget. In contrast, FedLAMA effectively increases the aggregation interval while maintaining the
convergence speed.

Figure 5: The learning curves of CIFAR-100 (WideResNet28-10) training (128 clients). a): The
curves for IID data distribution. b): The curves for non-IID data distribution (α = 0.1). FedAvg
(x) indicates FedAvg with the interval of x. FedLAMA (x, y) indicates FedLAMA with the base
interval of x and the interval increase factor of y. While FedAvg signiﬁcantly loses the convergence
speed as the aggregation interval increases, FedLAMA has a marginl impact on it which results in a
higher validation accuracy.

A.2 ADDITIONAL EXPERIMENTAL RESULTS

In this section, we provide extra experimental results with extensive hyper-parameter settings. We
commonly use 128 clients and a local batch size of 32 in all the experiments. The gradual learning
rate warmup (Goyal et al. (2017)) is also applied to the ﬁrst 10 epochs in all the experiments. Overall,
the learning curve charts and the validation accuracy tables deliver the key insight that FedLAMA
achieves a comparable convergence speed to the periodic full aggregation with the base interval

25

01002003000.20.40.60.81.0Validation accuracy (%)Epoch FedAvg (6) FedAvg (12) FedAvg (24)01002003000.20.40.60.81.0Validation accuracy (%)Epoch FedAvg (6) FedLAMA (6, 2) FedLAMA (6, 4)01002003001.01.52.02.5Training loss (softmax)Epoch FedAvg (6) FedLAMA (6, 2) FedLAMA (6, 4)01002003001.01.52.02.5Training loss (softmax)Epoch FedAvg (6) FedAvg (12) FedAvg (24)02000400060000.20.40.60.81.0Validation accuracy (%)Iteration FedAvg (6) FedAvg (12) FedAvg (24)02000400060000.20.40.60.81.0Validation accuracy (%)Iteration FedAvg (6) FedLAMA (6, 2) FedLAMA (6, 4)02000400060001.01.52.02.5Training loss (softmax)Iteration FedAvg (6) FedLAMA (6, 2) FedLAMA (6, 4)02000400060001.01.52.02.5Training loss (softmax)Iteration FedAvg (6) FedAvg (12) FedAvg (24)a) CIFAR-10 IID setting learning curvesb) CIFAR-10 non-IID setting learning curves01002001234Training loss (softmax)Epoch FedAvg (6) FedLAMA (6, 2) FedLAMA (6, 4)01002001234Training loss (softmax)Epoch FedAvg (6) FedAvg (12) FedAvg (24)01002000.20.40.60.8Validation accuracy (%)Epoch FedAvg (6) FedLAMA (6, 2) FedLAMA (6, 4)01002000.20.40.60.8Validation accuracy (%)Epoch FedAvg (6) FedAvg (12) FedAvg (24)02000400060000.20.40.60.8Validation accuracy (%)Iteration FedAvg (6) FedAvg (12) FedAvg (24)02000400060000.20.40.60.8Validation accuracy (%)Iteration FedAvg (6) FedLAMA (6, 2) FedLAMA (6, 4)02000400060001234Training loss (softmax)Iteration FedAvg (6) FedLAMA (6, 2) FedLAMA (6, 4)02000400060001234Training loss (softmax)Iteration FedAvg (6) FedAvg (12) FedAvg (24)a) CIFAR-100 IID setting learning curvesb) CIFAR-100 non-IID setting learning curvesUnder review as a conference paper at ICLR 2022

Figure 6: The learning curves of FEMNIST (CNN) training. FedAvg (x) indicates FedAvg with
the interval of x. FedLAMA (x, y) indicates FedLAMA with the base interval of x and the interval
increase factor of y. FedLAMA curves are not strongly affected by the increased aggregation interval
while FedAvg signiﬁcantly loses the convergence speed as well as the validation accuracy.

(τ ’) while having the communication cost that is similar to the periodic full aggregation with the
increased interval (φτ (cid:48)).

Artiﬁcial Data Heterogeneity – For CIFAR-10 and CIFAR-100, we artiﬁcially generate the het-
erogeneous data distribution using Dirichlet’s distribution. The concentration coefﬁcient α is set to
0.1, 0.5, and 1.0 to evaluate the performance of FedLAMA across a variety of degree of data hetero-
geneity. Note that the small concentration coefﬁcient represents the highly heterogeneous numbers
of local samples across clients as well as the balance of the samples across the labels. We used the
data distribution source code provided by FedML (He et al. (2020)).

CIFAR-10 – Figure 4 shows the full learning curves for IID and non-IID CIFAR-10 datasets. The
hyper-parameter settings correspond to Table 4 and 1. First, as the aggregation interval increases
from 6 to 24, FedAvg suffers from the slower convergence, and it results in achieving a lower val-
In contrast, FedLAMA learning curves are
idation accuracy, regardless of the data distribution.
marginally affected by the increased aggregation interval. Table 6 and 7 show the CIFAR-10 classi-
ﬁcation performance of FedLAMA across different φ settings. As expected, the accuracy is reduced
as φ increases. The IID and non-IID data settings show the common trend. Depending on the system
network bandwidth, φ can be tuned to be an appropriate value. When φ = 2, the accuracy is almost
the same as or even slightly higher than FedAvg accuracy. If the network bandwidth is limited, one
can increase φ and slightly increase the epoch budget to achieve a good accuracy. Table 8 shows the
CIFAR-10 accuracy across different τ (cid:48) settings. We see that the accuracy is signiﬁcantly dropped as
τ (cid:48) increases.

CIFAR-100 – Figure 5 shows the learning curves for IID and non-IID CIFAR-100 datasets. Likely
to CIFAR-10 results, FedAvg learning curves are strongly affected as the aggregation interval in-
creases from 6 to 24 while FedLAMA learning curves are not strongly affected. Table 9 and 10
show the CIFAR-100 classiﬁcation performance of FedLAMA across different φ settings. Fed-
LAMA achieves a comparable accuracy to FedAvg with a short aggregation interval, even when
the degree of data heterogeneity is extreamly high (25% device sampling and Direchlet’s coefﬁcient
of 0.1). Table 11 shows the FedAvg accuracy with different τ (cid:48) settings. Under the strongly het-
erogeneous data distributions, FedAvg with a large aggregation interval (τ ≥ 12) do not achieve a
reasonable accuracy.

FEMNIST – Figure 6 shows the learning curves of CNN training. Likely to the previous two
datasets, the periodic full aggregation suffers from the slower convergence as the aggregation in-
terval increases. FedLAMA learning curves are not much affected by the increased aggregation
interval, and it results in achieving a higher validation accuracy after the same number of iterations.
Table 12 shows the FEMNIST classiﬁcation performance of FedLAMA across different φ settings.
FedLAMA achieves a similar accuracy to the baseline (FedAvg with τ (cid:48) = 10) even when using a
large interval increase factor φ ≥ 4. These results demonstrate the effectiveness of the proposed
layer-wise adaptive model aggregation method on the problems with heterogeneous data distribu-
tions.

26

05001000150020000.20.40.60.81.0Validation accuracy (%)Iteration FedAvg (10) FedLAMA (10, 2) FedLAMA (10, 4)05001000150020000.20.40.60.81.0Validation accuracy (%)Iteration FedAvg (10) FedAvg (20) FedAvg (40)05001000150020001234Training loss (softmax)Iteration FedAvg (10) FedAvg (20) FedAvg (40)05001000150020001234Training loss (softmax)Iteration FedAvg (10) FedLAMA (10, 2) FedLAMA (10, 4)Under review as a conference paper at ICLR 2022

Table 6: (IID data) CIFAR-10 classiﬁcation results of FedLAMA with different φ settings.

# of clients

Local batch size

128

32

LR
0.8

0.5

Averaging interval: τ (cid:48)

6

Interval increase factor: φ
1 (FedAvg)
2
4
8

Validation acc.
88.37 ± 0.1%
88.41 ± 0.04%
86.33 ± 0.2%
85.08 ± 0.04%

Table 7: (Non-IID data) CIFAR-10 classiﬁcation results of FedLAMA with different φ settings.

# of clients

Local batch size

LR

τ (cid:48)

Active ratio

Dirichlet coeff.

φ

100%

100%

100%

50%

0.8

128

32

6

50%

50%

25%

25%

25%

0.6

0.3

1

0.5

0.1

1

0.5

0.1

1

0.5

0.1

1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4

Validation acc.
90.79 ± 0.1%
89.01 ± 0.04%
87.84 ± 0.01%
90.53 ± 0.18%
89.21 ± 0.2%
86.68 ± 0.12%
89.52 ± 0.11%
89.00 ± 0.1%
84.82 ± 0.08%
90.34 ± 0.12%
89.56 ± 0.13%
87.48 ± 0.21%
89.86 ± 0.13%
88.44 ± 0.15%
87.29 ± 0.18%
87.83 ± 0.2%
87.40 ± 0.17%
85.92 ± 0.21%
88.97 ± 0.03%
87.89 ± 0.2%
86.61 ± 0.1%
87.59 ± 0.05%
87.12 ± 0.08%
86.57 ± 0.02%
84.02 ± 0.04%
83.55 ± 0.02%
83.06 ± 0.03%

Table 8: (Non-IID data) CIFAR-10 classiﬁcation results of FedAvg with different τ (cid:48) settings.

# of clients

Local batch size

LR

128

128

32

32

0.8

0.3

τ (cid:48)
6
12
24
6
12
24

Active ratio

Dirichlet coeff.

φ

100%

25%

0.1

0.1

1 (FedAvg)
1 (FedAvg)
1 (FedAvg)
1 (FedAvg)
1 (FedAvg)
1 (FedAvg)

Validation acc.
89.52 ± 0.11%
87.29 ± 0.05%
84.82 ± 0.1%
84.02 ± 0.1%
82.48 ± 0.2%
76.72 ± 0.1%

Table 9: (IID data) CIFAR-100 classiﬁcation results of FedLAMA with different φ settings.

# of clients

Local batch size

LR

Averaging interval: τ (cid:48)

128

32

0.6

6

Interval increase factor: φ
1 (FedAvg)
2
4
8

Validation acc.
76.50 ± 0.02%
75.99 ± 0.03%
76.17 ± 0.2%
76.15 ± 0.2%

27

Under review as a conference paper at ICLR 2022

Table 10: (Non-IID data) CIFAR-100 classiﬁcation results of FedLAMA with different φ settings.

# of clients

Local batch size

LR

τ (cid:48)

Active ratio

Dirichlet coeff.

φ

128

32

0.4

100%

100%

0.2

100%

50%

6

50%

50%

25%

25%

25%

0.4

0.2

0.4

0.2

0.4

0.2

1

0.5

0.1

1

0.5

0.1

1

0.5

0.1

1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4
1 (FedAvg)
2
4

Validation acc.
80.34 ± 0.01%
78.92 ± 0.01%
77.16 ± 0.05%
80.19 ± 0.02%
78.88 ± 0.1%
78.03 ± 0.08%
79.78 ± 0.02%
79.07 ± 0.02%
79.32 ± 0.01%
79.94 ± 0.1%
78.98 ± 0.01%
77.50 ± 0.02%
79.95 ± 0.05%
78.37 ± 0.05%
76.93 ± 0.1%
79.62 ± 0.06%
78.76 ± 0.02%
77.44 ± 0.02%
78.78 ± 0.02%
78.10 ± 0.02%
76.84 ± 0.03%
78.81 ± 0.01%
77.86 ± 0.04%
77.01 ± 0.1%
79.06 ± 0.03%
78.63 ± 0.02%
77.17 ± 0.01%

Table 11: (Non-IID data) CIFAR-100 classiﬁcation results of FedAvg with different τ (cid:48) settings.

# of clients

Local batch size

LR

128

128

32

32

0.4

0.4

τ (cid:48)
6
12
24
6
12
24

Active ratio

Dirichlet coeff.

φ

100%

25%

0.1

0.1

1 (FedAvg)
1 (FedAvg)
1 (FedAvg)
1 (FedAvg)
1 (FedAvg)
1 (FedAvg)

Validation acc.
79.78 ± 0.02%
77.71 ± 0.1%
69.63 ± 0.1%
79.06 ± 0.03%
76.16 ± 0.05%
67.43 ± 0.1%

Table 12: FEMNIST classiﬁcation results of FedLAMA with different φ settings.

# of clients

Local batch size

LR

Averaging interval: τ (cid:48)

Active ratio

100%

128

32

0.04

12

50%

25%

28

Interval increase factor: φ
1 (FedAvg)
2
4
8
1 (FedAvg)
2
4
8
1 (FedAvg)
2
4
8

Validation acc.
85.74 ± 0.21%
85.40 ± 0.13%
84.67 ± 0.1%
84.15 ± 0.18%
86.59 ± 0.2%
86.07 ± 0.1%
85.77 ± 0.15%
85.31 ± 0.03%
86.04 ± 0.2%
86.01 ± 0.1%
85.62 ± 0.08%
85.23 ± 0.1%

