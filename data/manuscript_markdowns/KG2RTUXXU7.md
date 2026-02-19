Local K-means: An Efﬁcient Optimization Algorithm
And Its Generalization

Anonymous Author(s)
Afﬁliation
Address
email

Abstract

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

Until now, k-means is still one of the most popular clustering algorithms because of
its simplicity and efﬁciency, although it has been proposed for a long time. In this
paper, we considered a variant of k-means that takes the k-nearest neighbor (k-NN)
graph as input and proposed a novel clustering algorithm called Local K-Means
(LKM). We also developed a general model that uniﬁed LKM, KSUMS, and SC,
and discussed the connection among them. In addition, we proposed an efﬁcient
optimization algorithm for the uniﬁed model. Thus, not only LKM but also SC can
be optimized with a linear time complexity with respect to the number of samples.
Speciﬁcally, the computational overhead is O(nk), where n and k are denote the
number of samples and nearest neighbors, respectively. Extensive experiments
have been conducted on 11 synthetic and 16 benchmark datasets from the literature.
The effectiveness, efﬁciency, and robustness to outliers of the proposed method
have been veriﬁed by the experimental results.

1

Introduction

Clustering is one of the fundamental tasks of machine learning [10]. It plays a very important role in
many applications such as document analysis [6], image processing [14], and recommender system
[12]. Given a dataset with n samples and the number of clusters c, its purpose is to split these samples
into c disjoint groups, so that the samples within the same group are similar to each other, and the
samples between different groups are not. Although there are lots of clustering algorithms have been
proposed, k-means is still getting a lot of attention. In this paper, we proposed an efﬁcient clustering
method called local k-means where a k-NN graph is taken as input. It can be seen as a variant of
traditional k-means. In the following, the two basic materials of our model are ﬁrstly described, and
the main contributions of this article will be mentioned at the end of this section.

Notations: Bold capital letters and bold lowercase letters denote matrices and vectors, respectively.
The symbols n, d, and c are respectively used to represent the number of samples of the dataset, the
number of features, and the number of clusters to construct. For matrix A, we call it indicator matrix,
if each row of it has only one element equal to 1. Φn×c is the set of all indicator matrices.

1.1 k-means

As one of the most popular clustering algorithms, k-means aims to group n samples into c clusters
where each sample belongs to the cluster with the nearest cluster centers. Let X = [x1, · · · , xn]T ∈
Rn×d be a collection of samples to cluster, where xi ∈ Rd denotes the i-th sample. Then the objective
function of k-means can be formulated as

min
A1,··· ,Ac

c
(cid:88)

(cid:88)

k=1

xi∈Ak

(cid:107)xi − mk(cid:107)2
2,

(1)

Submitted to 35th Conference on Neural Information Processing Systems (NeurIPS 2021). Do not distribute.

: User

B

D

A

C

Figure 1: Community in the social network. There is a connection between two users if they know
each other, in other words, the two people are friends with each other. The thicker the line, the
more familiar the two users. According to the connections between users, the clustering algorithm
divides them into disjoint sets. For example, a partition composed of A, B, C, and D is a satisfactory
clustering result.

33

34

35

36

37

where Ak denotes the set of samples in the i-th cluster, A1
mk denotes the mean of samples in Ak.
Although the problem in Eq. (1) is computationally difﬁcult, 1 many efﬁcient optimization algorithms
where a local optimum will be found quickly have been proposed. Among them, Lloyd’s algorithm is
the most widely used. Let Y = [y1, · · · , yn]T = [¯y1, · · · , ¯yc] ∈ Rn×c be an indicator matrix, i.e.,

(cid:83) · · · (cid:83) Ac = {xi | i = 1, · · · , n}, and

yij =

(cid:26) 1
0

xi ∈ Aj
otherwise

, i = 1, · · · , n, j = 1, · · · , c,

38

the problem in Eq. (1) can be then rewritten as

min
Y

(cid:107)X − YM(cid:107)2
2,

(2)

(3)

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

where M = (YT Y)−1YT X. In Lloyd’s algorithm, Y and M are regarded as two independent
variables and be optimized alternately.

1.2 Data in the form of graph

In ﬁelds such as social networks and recommendation systems, the data being studied is often
presented in the form of graphs. In other words, for a single sample, we have no features to describe
it, what we have is only the relationship between it and others, as shown in Figure 1.
In generally, a sparse similarity matrix W ∈ Rn×n can be used to describe this kind of data, i.e.,

wij =

(cid:26) f (xi, xj)
0

If xi and xj are directly connected
Otherwise

, i, j = 1, · · · , n,

(4)

where f (xi, xj) represents the similarity between xi and xj, and its value can be usually obtained
directly.

Based on the above discussion, a k-means-like algorithm is proposed, which takes the k-NN graph
as input and can be quickly optimized. In addition, we also discussed its connection with other
algorithms, such as KSUMS and spectral clustering. Here, we summarize the main contributions of
the article as follows

• A novel clustering algorithm called Local K-Means (LKM) is proposed. Because only the
distances between the sample and its neighbors are considered, LKM is robust to outliers.
• The relationship between LKM and other algorithms (KSUMS and SC) is discussed, and a

uniﬁed model is established.

• An efﬁcient optimization algorithm for the uniﬁed model is developed, from which we ﬁnd
that the spectral clustering model can be optimized in the same way as LKM, which means
both of them can also be optimized in O(nk) time.

1Speciﬁcally, it is an np-hard problem.

2

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

2 Related work

A disadvantage of k-means is that its performance will be affected largely by the initialization of
the cluster center. To this end, a lot of efforts have been made, such as [2, 4, 3]. In these methods,
the cluster center is carefully initialized through a special process. In addition to the more robust
clustering result, an improvement of performance can also be achieved. More related work can be
found here [15, 22].

Since the computational complexity of k-means involves the product of the number of samples
and clusters, it will be very time-consuming if the two numbers are very large. With the help of
techniques that used to accelerate the nearest neighbor search, the nearest center for each sample
can be quickly found without computing distances to all centers [25, 11]. [7] developed a fast
implementation of k-means using coreset. A partition on a small coreset is computed ﬁrstly and is
used as an initialization on a larger coreset. In [32], Xia et al. described each cluster by a ball and
proposed Ball k-means which accelerated k-means by reducing the computation of distances between
samples and centers. [13] proposed compressive k-means (CKM) where the centers are estimated
from a sketch (a compressed representation of the original data). Once the sketch is obtained, the
computational overhead is then independent of the size of the original data. Moreover, it’s also a hot
spot to use the advantages of GPU to shorten the time consumed by k-means, such as [17] and [5].

Clustering on graph data is also a hot topic. Some well-known algorithms include [19, 29, 21].
However, these algorithms often have a time complexity that increases quadratically with respect to
the number of samples. To this end, many fast versions of them are proposed [33, 20, 9].

3 The proposed model

In our article, how to solve the problem in Eq. (1) has not been paid attention to, but some simple
derivations are ﬁrstly made on it. Therefore we can analyze the meaning of the problem from the
perspective of a distance graph. For convenience, we deﬁne Nk(xi) = {xj | xj is among the
k-nearest neighbors of xi or xi is among the k-nearest neighbors of xj}, and start from the following
equivalent form of k-means

min
A1,··· ,Ac

c
(cid:88)

k=1

1
|Ak|

(cid:88)

(cid:107)xi − xj(cid:107)2
2,

xi,xj ∈Ak

(5)

85

With the help of the deﬁnition of Y in Eq. (2), problem (5) can be equivalently expressed as follows

min
Y∈Φn×c

⇔ min

Y∈Φn×c

diag (cid:0)(YT Y)−1(cid:1)T

diag (cid:0)YT DY(cid:1) ,

T r (cid:0)(YT Y)−1YT DY(cid:1) ,

(6)

(7)

86

87

where diag(A) = [a11, · · · , ann]T . Obviously, if we only consider the distances between the sample
and its neighbors, then the problem in Eq. (7) can be expressed as
(YT Y)−1YT D(k)Y

T r

(cid:16)

(cid:17)

(8)

,

min
Y∈Φn×c

88

with

d(k)

ij =

(cid:26) (cid:107)xi − xj(cid:107)2
γ

2

if xi ∈ Nk(xj)
Otherwise

,

(9)

89

90

91

92

93

94

95

where γ is the maximum value of set {(cid:107)xi − xj(cid:107)2
is the ﬁnal objective function of LKM.

2 | xi ∈ Nk(xj), i = 1, · · · , n}. The Equation (8)

From the discussion in Section 1.2, we know that only the similarity instead of the distance between
samples can be obtained directly in graph data. Fortunately, in practical applications, we can convert
the similarity to dissimilarity by

rij =

(cid:26) −log(sij)
β

0 < sij
sij = 0

,

(10)

where sij is the normalized2 similarity between xi and xj, β is the maximum value of set {−log(sij) |
i, j = 1, · · · , n}. Then the dissimilarity can be used to replace the distance in the model.

2sij ∈ [0, 1]

3

96

97

98

99

100

101

102

103

104

105

106

107

108

3.1 Generalization

It is not difﬁcult to ﬁnd that LKM, KSUMS [23], and Ratio-cut [29] can all be represented uniformly
by the following model

(cid:16)

min
Y∈Φn×c

T r

(YT Y)−pYT G(k)Y

,

(11)

(cid:17)

where g(k)
meaning of p will be explored in future work.

ij denotes the dissimilarity or distance between xi and xj, and p >= 0 is a parameter. The

Instances of KSUMS and LKM: The objective function of KSUMS is

min
Y∈Φn×c

T r

(cid:16)

YT D(k)Y

(cid:17)

,

(12)

where D(k) takes the same expression as that in LKM. Let g(k)
is identical with KSUMS (12) if p = 0, and is identical with LKM if p = 1.

ij be setted by Eq. (9), the problem (11)

Instance of Ratio-cut: Beneﬁting from the introduction of Y, the problem of ratio-cut (an algorithm
that belongs to the spectral clustering (SC) family) can be expressed as
T r (cid:0)(YT Y)−1YT (∆ − W)Y(cid:1) ,

(13)

min
Y∈Φn×c

where ∆ is a diagonal matrix, ∆ii = (cid:80)n
determined by heat kernel, i.e., wij = e−
problem (11) is equivalent with ratio-cut, if p = 1 and g(k)
ij

(cid:107)xi−xj (cid:107)2
2
t

j=1 wij. In generally, the similarity matrix W can be

if xi ∈ Nk(xj), wij = 0 otherwise. Therefore the

is setted by

g(k)
ij =






(cid:80)n

j=1 wij
−wij
0

i = j
i (cid:54)= j, and xi ∈ Nk(xj)
Otherwise

.

(14)

109

3.2 Optimization

110

111

112

From the discussion above, we know that the problem of LKM can be expressed by Eq. (11) with
p = 1. Therefore, an optimization algorithm for problem (11) instead of problem (8) is developed.
To begin with, some notations are presented as follows

si (cid:44) ¯yT
ni (cid:44) ¯yT

i G(k) ¯yi,
i ¯yi,

i = 1, · · · , c,

i = 1, · · · , c,

113

the problem (11) then becomes

min
Y∈Φn×c

Obj(Y), with Obj(Y) =

c
(cid:88)

i=1

si
np
i

.

(15)

(16)

(17)

114

115

In the following derivation, the i-th row of Y (i.e., yi) is regarded as the variable to be optimized
while others are ﬁxed, and yi = eα before updated. Thus yi can be updated by

yi = eβ,

β = arg min

j

Obj(yi = ej) − Obj(yi = 0),

(18)

where ei = [0, · · · , 1, · · · , 0] be a vector with all elements equal to 0, except the i-th, which is 1, and
0 is the column vector of all zeros,

Because Obj(yi = 0) is constant, the above formula holds. According to Eq. (17), we have



j (cid:54)= α

sj +bj

Obj(yi = ej) − Obj(yi = 0) =

, j = 1, · · · , c,

j = α

(nj +1)p − sj
np
j
− sj −bj
sj
np
(nj −1)p
j



116

117

118

119

with

(cid:40)

bj =

2 (cid:80)
2 (cid:80)

xl∈Aj

xl∈Aj

g(k)
il + g(k)
g(k)
il − g(k)

ii

ii

j (cid:54)= α

j = α

,

4

(19)

(20)

Algorithm 1: An efﬁcient program for solving problem (11).
Note: The vector y ∈ Rn denotes the clustering result, i.e., yi is the cluster that xi belongs to.
The Eq. (15), (16), and (20) involved in the algorithm have high computational complexity, but
these can be computed more efﬁciently if the sparsity of G(k) is considered. See the
supplementary material for a more detailed algorithm;
Data: Sparse matrix 3G(k) ∈ Rn×n, the number of cluster c
Result: The clustering result y
Initialize y randomly;
Compute vector s and n by Eq. (15) and (16), respectively;
while not converge do

for i = 1, · · · , n do

Compute bj by Eq. (20) for j ∈ Bi;
Compute Obj(yi = j) − Obj(yi = 0) by Eq. (19) for j ∈ Bi;
Update yi by Eq. (18);
Update s and n by Eq. (21) and Eq. (22), respectively;

120

121

122

123

124

Beneﬁting from the sparsity of G(k), it takes O(nk), O(k + c), and O(k) time to compute s, b, and
n, respectively. Therefore, the proposed optimization algorithm has a computational complexity of
O(n2k + nc), which is unbearable, for large-scale datasets. However, if the variables s and n are
computed in advance and updated following the update of yi, then the computational complexity of
the algorithm can greatly be reduced. The update rules for s and n are as follows
sα ⇐ sα − bα,
sβ ⇐ sβ + bβ,
nα ⇐ nα − 1, nβ ⇐ nβ + 1,

(21)
(22)

125

Thus, the computational complexity of the optimization algorithm is O(n(k + c)).

126

127

128

129

130

131

132

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

On more step From Eq. (11), we know that only the information of pair (xi, xj) is considered in
the model, and there are at most 2nk such pairs. For convenience, we assume that there are exactly
2k such pairs for each sample xi, i.e., 2k = |{(xi, xj) | xj ∈ Nk(xi) or xi ∈ Nk(xj)}|. For cluster
j, we call it an element of Bi (j ∈ Bi), if there is at least one sample in cluster j belongs to Nk(xi)
or xi belongs to the set of neighbors of these samples. Based on the assumption and notations above,
we know that when updating yi by Eq. (18), the size of Bi is at most 2k. However, it does not make
sense to group the sample xi into cluster j (cid:54)∈ Bi, from the perspective of the performance. Therefore,
we only need to pay attention to the cases where j ∈ Bi. Thus, the computational complexity of the
optimization algorithm can be reduced to O(nk).

Time and space complexity From Algorithm 1, we can see that the memory is mainly occupied
by the matrix G(k) ∈ Rn×n, which is equivalent to a sparse matrix, and contains at most 2nk
non-constants. The memory overhead caused by other variables is O(n) at most. For example, y,
Bi, and s require O(n), O(k), and O(c) memory, respectively. Thus the memory overhead of LKM
is O(nk). Beneﬁting from the sparsity of G(k), Eq. (15), (16), and (20) can all be calculated more
efﬁciently. Speciﬁcally, only O(nk), O(n), and O(k) time are needed respectively, please refer to the
supplementary materials for details. After yi is updated, only O(1) time is needed to update variables
s and n. Thus, the computational complexity of LKM is O(nk).

4 Experiments

In this section, the performance of the proposed algorithm, LKM, is veriﬁed on eleven synthetic
datasets and sixteen benchmark datasets. The rest of this section is organized as follows: First,
experiments on synthetic datasets are shown. In short, Mickey, Outlier, and family of Grid datasets
are used to verify the effectiveness, robustness, and efﬁciency of LKM, respectively. Then, we
compare 7 popular clustering algorithms with LKM on 16 benchmark datasets, to evaluate the
performance of the proposed algorithm.

3Strictly speaking, G(k) is not a sparse matrix. However, at most 2nk values in G(k) are not equal to λ, so it

can be regarded as a sparse matrix.

5

150

4.1 Experiments conducted on synthetic datasets

151

152

153

154

155

156

157

158

159

Experiment on “Mickey” To verify the effectiveness of LKM, a synthetic dataset called “Mickey”
is constructed. The distribution of points is shown in Figure 2(a). The triangles representing the
means of the clusters are not points of the datasets.

From Figure 2(b) and 2(c), we found that The proposed method LKM successfully found the
cluster structure, but k-means did not. k-means still cannot ﬁnd the correct structure, even with the
initialization of the ground truth label. Because the distance between point 1 and the blue triangle
(mean of all blue points), d1 is greater than the distance between point 1 and the orange triangle (mean
of all orange points), d2, k-means will group it into the blue cluster instead of orange. Therefore,
k-means cannot handle datasets like this.

(a) Original

(b) k-means

(c) LKM

Figure 2: The performance of k-means and LKM on “Mickey”.

160

161

162

163

164

165

166

167

168

169

Experiment on “Outlier”
In order to verify the robustness of our method, we construct a dataset
called “Outlier”. It consists of four clusters with centers (0, 0), (0, 5), (5, 0), and (5, 5), and an outlier
with the coordinate of (100, 100). The distance between outlier A and other points is not as close as
shown in Figure 3. From Figure 3(b) and 3(c), we can see that the performance of k-means is severely
affected by the outlier A, while the performance of LKM is not. In k-means, the center of the cluster
containing abnormal points will largely shift towards the direction of the abnormal points, resulting
in poor performance. In LKM, the distance between xi and xj is not calculated if xj (cid:54)∈ Nk(xi), but
a parameter λ is used instead, so ideally, the distance between any two points belonging to different
clusters is λ. In other words, for the sample point xi, there is no difference between the outlier and
the samples that do not belong to Nk(xi).

(a) Original

(b) k-means

(c) LKM

Figure 3: The performance of k-means and LKM on “Outlier”.

170

171

172

173

174

175

176

Experiments on the family of “Grid” In order to verify the efﬁciency of LKM, in this paragraph,
9 synthetic datasets called Toy-1, Toy-2, · · · , Toy-9 are constructed. These datasets share the same
structure, and their distributions are similar to that shown in Figure 4. In these datasets, each cluster
is always composed of 10 points generated by Gaussian distribution. Since the time complexity of
LKM and k-means is closely related to the number of points, we set different sizes for these data
sets, ranging from 1960 to 125440. The number of clusters and the standard deviation involved in the
Gaussian distribution for each dataset is shown in Table 1.

6

(a) k-means

(b) LKM

Figure 4: The performance of k-means and LKM on Toy-1.

Table 1: Performance of k-means and LKM

Datasets

# Clusters

Toy-1
Toy-2
Toy-3
Toy-4
Toy-5
Toy-6
Toy-7
Toy-8
Toy-9

196
196
196
3136
3136
3136
12544
12544
12544

Precision

Recall

F1 score

k-means LKM k-means LKM k-means LKM

0.854
0.834
0.785
0.856
0.832
0.783
0.855
0.833
0.785

0.975
0.948
0.874
0.981
0.947
0.883
0.982
0.948
0.884

0.915
0.885
0.828
0.918
0.881
0.825
0.917
0.882
0.826

0.983
0.957
0.889
0.988
0.957
0.893
0.988
0.957
0.896

0.883
0.859
0.806
0.886
0.856
0.803
0.885
0.857
0.805

0.979
0.953
0.881
0.984
0.952
0.888
0.985
0.952
0.890

3σ

0.5
0.6
0.7
0.5
0.6
0.7
0.5
0.6
0.7

Table 2: Time (s) consumed by k-means and LKM
k-means

FLK

Datasets Ball-Tree

Algo. 1

# Iter.

Total

# Iter.

Total

Speed-up

Toy-1
Toy-2
Toy-3
Toy-4
Toy-5
Toy-6
Toy-7
Toy-8
Toy-9

6.26E-03
6.54E-03
6.27E-03
1.34E-01
1.37E-01
1.39E-01
6.50E-01
6.04E-01
6.18E-01

1.30E-03
1.66E-03
1.73E-03
2.64E-02
3.32E-02
3.98E-02
1.35E-01
1.64E-01
1.95E-01

3.96
5.66
5.96
5.80
7.64
9.40
7.20
9.08
10.96

7.56E-03
8.20E-03
8.00E-03
1.60E-01
1.70E-01
1.79E-01
7.85E-01
7.68E-01
8.13E-01

13.12
14.32
15.32
14.68
16.62
18.50
16.22
17.58
18.88

5.97E-03
5.57E-03
6.00E-03
2.00E+00
2.27E+00
2.55E+00
3.89E+01
4.21E+01
4.50E+01

1.39E+00
1.33E+00
1.35E+00
3.00E+01
3.15E+01
3.25E+01
1.28E+02
1.33E+02
1.34E+02

177

178

179

180

181

182

183

184

185

186

In Table 2, the column named “Ball-Tree” represents the time it takes to construct the graph required
by LKM through Ball-tree with k = 20. The column named “# Iter” denotes the number of iterations
required for the algorithm to converge. The total time of LKM refers to the sum of the time consumed
by Ball-Tree and Algorithm 1. The speed-up is the ratio of the time consumed by each iteration of
k-means to the time consumed by each iteration of Algorithm 1. Both k-means and LKM were run
50 times, and the average results were reported.

As shown in Table 2, Algorithm 1 consumes a signiﬁcantly shorter time than k-means, which is more
obvious on datasets with more clusters. The main reason is that when yi is going to update, only the
case where j ∈ Bi is considered. In addition, LKM has a signiﬁcant improvement in terms of the
quality of the clustering result, compared to k-means, as shown in Table 1 and Figure 4.

7

187

4.2 Experiments conducted on benchmark datasets

188

189

190

191

192

193

194

195

196

197

198

199

200

201

202

203

204

205

206

207

208

209

210

211

212

213

214

215

216

217

218

219

220

221

222

223

224

225

226

227

228

229

230

231

232

233

234

235

236

4.2.1 Datasets

Sixteen benchmark datasets are used including LFW [8], CPLFW [34], CALFW [35], FERET [24],
Colon [1], MUCT [18], CMUPIE [30], CFPW [27], Dexter, Madelon, GTDB, FaceV5, Mpeg7,
Olivetti, Yale, and Umist. All facial datasets are processed by the way [23]. For those non-facial
datasets, PCA [31] is adopted and some components are selected such that the amount of variance is
greater than 95% if the dimensionality of the datasets is larger than 1024. The names of datasets are
all linked to where the dataset can be download. The introduction to these datasets can be found in
the supplemental material.

4.2.2 Baselines and experimental settings

We compare LKM with several clustering algorithms, including AGCI [33], FINCH [26], k-means
[16], KSUMS [23], RCC [28], SC [29], and FCDMF [20]. For graph-based methods, i.e., KSUMS,
RCC, and SC, the number of nearest neighbors, k, is ﬁxed at 20. For anchor-based methods, AGCI
and FCDMF, the number of anchors is always set by m = min(n/2, 1024). Whether k-NN graph
or anchor graph, heat-kernel is always adopted to construct the graph. In FINCH, we take the
clustering result with the number of clusters closest to the number of ground truth clusters as the
ﬁnal clustering result. In RCC, the threshold to assign points together in a cluster is tuned from
{0.1, 0.3, 0.5, 0.7, 0.9}. K-means is initialized in a random way and the step of k-means involved in
AGCI and SC share the same conﬁguration with k-means itself. If the performance of the algorithm
is related to the initialization, we run it repeatedly 50 times and report the average performance.

We run all methods on an Arch machine with i7-8700 CPU (3.20 GHz), 32 GB main memory.

4.2.3 Experimental results

Clustering ACCuracy (ACC), Normalized Mutual Information (NMI), and Adjusted Rand index
(ARI) are used to evaluate the performance of these algorithms. From Table 3, we can clearly see that:
(1) In most cases LKM has achieved the highest performance comparing to several state-of-the-art
algorithms, which veriﬁed the effectiveness of the proposed algorithm. Speciﬁcally, LKM exceeds
the second-best results 24.4%, 4.6%, 4.8%, 1.5% and 1.3% on CALFW, LFW, Umist, Olivetti, and
CMU respectively, in terms of ACC. Under the metrics of NMI and ARI, we can come to similar
results. (2) Although only slight improvements LKM has achieved over many datasets compared
to the second-best results, the computational complexity of LKM is much lower than that of most
algorithms, which is an important property of LKM. (3) RCC has poor performance on FaceV5,
CMU, GTdb, Umist, and Yale, which may be caused largely by an inappropriate threshold, while
only one parameter (the number of neighbors) is needed in LKM, is an integer and easy to tune. In
addition, the inﬂuence of parameter k (the number of neighbors) on clustering performance has been
studied, and the results are shown in the supplemental material.

5 Conclusions

In this paper, we devote ourselves to an unsupervised learning problem, clustering. An efﬁcient
clustering algorithm called Local K-Means (LKM) was proposed. It can be seen as a variant of
k-means that takes the k-NN graph as input. We also discussed a general model that uniﬁed LKM,
KSUMS, and SC. Thus the connection among them can be easily established. In addition, we
developed an efﬁcient optimization algorithm for the uniﬁed model, so that not only LKM but also
SC can be optimized in O(nk) time, which is very important for large-scale datasets, especially for
these datasets with a large number of clusters. In order to verify the advantages of LKM, extensive
experiments on eleven synthetic and sixteen benchmark datasets are conducted, and the results have
shown the effectiveness, efﬁciency, and robustness of our model.

In some cases where k-NN graphs are not available, our algorithm cannot work,
Limitations
in other words, a graph construction algorithm is necessary. Although many methods have been
proposed, it is still very difﬁcult to effectively construct an approximate k-NN graph if the number of
features is large. Thus, in these situations, the graph construction algorithm will produce a k-NN
graph of poor quality that would lead to poor performance of clustering results.

8

Datasets Met. AGCI

FCDMF

FIN

k-means KSUMS RCC

SC

Table 3: Performance on benchmark datasets

LFW

CALFW

CPLFW

FaceV5

CFPW

CMU

Colon

Dexter

FERET

GTdb

Madelon

Mpeg7

MUCT

Olivetti

Umist

Yale

ACC 0.460
0.866
NMI
0.063
ARI

ACC 0.599
0.887
NMI
0.187
ARI

ACC 0.537
0.770
NMI
0.209
ARI

ACC 0.730
0.930
NMI
0.605
ARI

ACC 0.537
0.770
NMI
0.209
ARI

ACC 0.185
0.409
NMI
0.079
ARI

ACC 0.690
0.178
NMI
0.208
ARI

ACC 0.579
0.077
NMI
0.035
ARI

ACC 0.522
0.822
NMI
0.354
ARI

ACC 0.454
0.658
NMI
0.313
ARI

ACC 0.517
0.003
NMI
0.004
ARI

ACC 0.463
0.660
NMI
0.278
ARI

ACC 0.732
0.928
NMI
0.612
ARI

ACC 0.509
0.722
NMI
0.366
ARI

ACC 0.413
0.626
NMI
0.320
ARI

ACC 0.395
0.448
NMI
0.187
ARI

0.450
0.860
0.078

0.399
0.859
0.084

0.355
0.689
0.167

0.517
0.829
0.280

0.355
0.689
0.167

0.154
0.372
0.063

0.581
0.010
0.011

0.627
0.124
0.063

0.378
0.734
0.211

0.419
0.634
0.282

0.513
0.001
0.000

0.445
0.650
0.295

0.741
0.922
0.698

0.407
0.643
0.263

0.412
0.589
0.300

0.344
0.398
0.139

0.460
0.866
0.063

0.599
0.888
0.190

0.546
0.772
0.208

0.731
0.931
0.621

0.546
0.772
0.208

0.182
0.407
0.077

0.608
0.094
0.078

0.596
0.091
0.042

0.521
0.822
0.353

0.459
0.661
0.319

0.521
0.005
0.006

0.462
0.666
0.291

0.722
0.923
0.586

0.510
0.718
0.366

0.416
0.628
0.317

0.397
0.455
0.196

0.373
0.711
0.008

0.504
0.696
0.007

0.584
0.613
0.012

0.535
0.829
0.290

0.584
0.613
0.012

0.165
0.306
0.018

0.629
0.129
0.249

0.153
0.080
0.011

0.495
0.686
0.039

0.391
0.579
0.211

0.456
0.001
0.000

0.442
0.617
0.153

0.972
0.991
0.971

0.480
0.674
0.323

0.468
0.673
0.375

0.339
0.358
0.119

9

0.454
0.850
0.037

0.419
0.878
0.098

0.738
0.889
0.627

0.934
0.979
0.899

0.738
0.889
0.627

0.286
0.571
0.192

0.635
0.108
0.110

0.584
0.024
0.031

0.546
0.839
0.439

0.533
0.690
0.382

0.529
0.005
0.006

0.539
0.720
0.414

0.982
0.992
0.976

0.569
0.758
0.443

0.450
0.641
0.355

0.443
0.495
0.234

0.551
0.805
0.592

0.573
0.886
0.373

0.745
0.857
0.201

0.069
0.105
0.001

0.745
0.858
0.202

0.015
0.000
0.000

0.581
0.045
-0.05

0.490
0.051
0.002

0.661
0.714
0.022

0.047
0.032
0.002

0.500
0.000
0.000

0.429
0.701
0.452

0.754
0.922
0.700

0.550
0.780
0.387

0.083
0.000
0.000

0.067
0.000
0.000

0.424
0.703
0.010

0.560
0.754
0.005

0.527
0.733
0.089

0.621
0.812
0.070

0.527
0.733
0.089

0.285
0.552
0.173

0.737
0.143
0.210

0.567
0.015
0.017

0.463
0.735
0.036

0.491
0.666
0.314

0.507
0.000
0.000

0.462
0.657
0.220

0.627
0.791
0.093

0.527
0.723
0.364

0.431
0.634
0.323

0.405
0.456
0.194

LKM

0.597
0.893
0.100

0.843
0.971
0.729

0.742
0.865
0.333

0.938
0.983
0.910

0.742
0.865
0.333

0.299
0.582
0.201

0.748
0.259
0.317

0.612
0.123
0.050

0.621
0.863
0.520

0.541
0.697
0.387

0.534
0.005
0.006

0.552
0.721
0.346

0.979
0.995
0.980

0.584
0.768
0.456

0.516
0.690
0.428

0.452
0.498
0.239

237

238

239

240

241

242

243

244

245

246

247

248

249

250

251

252

253

254

255

256

257

258

259

260

261

262

263

264

265

266

267

268

269

270

271

272

273

274

275

276

277

278

279

280

281

References

[1] U. Alon, N. Barkai, D. A. Notterman, K. Gish, S. Ybarra, D. Mack, and A. J. Levine. Broad
patterns of gene expression revealed by clustering analysis of tumor and normal colon tis-
sues probed by oligonucleotide arrays. Proceedings of the National Academy of Sciences,
96(12):6745–6750, 1999.

[2] D. Arthur and S. Vassilvitskii. k-means++: The advantages of careful seeding. Technical report,

Stanford, 2006.

[3] O. Bachem, M. Lucic, H. Hassani, and A. Krause. Fast and provably good seedings for k-means.
In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, Advances in
Neural Information Processing Systems 29, pages 55–63. Curran Associates, Inc., 2016.

[4] O. Bachem, M. Lucic, S.H. Hassani, and A. Krause. Approximate k-means++ in sublinear time.
In Proceedings of the Thirtieth AAAI Conference on Artiﬁcial Intelligence, pages 1459–1467,
2016.

[5] S. Cuomo, V. De Angelis, G. Farina, L. Marcellino, and G. Toraldo. A gpu-accelerated parallel

k-means algorithm. Computers & Electrical Engineering, 75:262–274, 2019.

[6] G.R. De Miranda, R. Pasti, and L.N. de Castro. Detecting topics in documents by clustering
word vectors. In International Symposium on Distributed Computing and Artiﬁcial Intelligence,
pages 235–243. Springer, 2019.

[7] G. Frahling and C. Sohler. A fast k-means implementation using coresets. International Journal

of Computational Geometry & Applications, 18(06):605–625, 2008.

[8] B.H. Gary, R. Manu, B. Tamara, and L.M.r Erik. Labeled faces in the wild: A database for
studying face recognition in unconstrained environments. Technical Report 07-49, University
of Massachusetts, Amherst, October 2007.

[9] L. He, N. Ray, Y. Guan, and H. Zhang. Fast large-scale spectral clustering via explicit feature

mapping. IEEE Transactions on Cybernetics, 49(3):1058–1071, 2019.

[10] A.K. Jain, M.N. Murty, and P.J. Flynn. Data clustering: a review. ACM computing surveys

(CSUR), 31(3):264–323, 1999.

[11] T. Kanungo, D.M. Mount, N.S. Netanyahu, C.D. Piatko, R. Silverman, and A.Y. Wu. An
efﬁcient k-means clustering algorithm: Analysis and implementation. IEEE transactions on
pattern analysis and machine intelligence, 24(7):881–892, 2002.

[12] R. Katarya and O.P. Verma. An effective web page recommender system with fuzzy c-mean

clustering. Multimedia Tools and Applications, 76(20):21481–21496, 2017.

[13] N. Keriven, N. Tremblay, Y. Traonmilin, and R. Gribonval. Compressive k-means. In 2017
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages
6369–6373, 2017.

[14] W. Kim, A. Kanezaki, and M. Tanaka. Unsupervised learning of image segmentation based on
differentiable feature clustering. IEEE Transactions on Image Processing, 29:8055–8068, 2020.

[15] M. Li, D. Xu, D. Zhang, and J. Zou. The seeding algorithms for spherical k-means clustering.

Journal of Global Optimization, pages 695–708, 2019.

[16] S. Lloyd. Least squares quantization in pcm.

IEEE transactions on information theory,

28(2):129–137, 1982.

[17] C. Lutz, S. Breß, T. Rabl, S. Zeuch, and V. Markl. Efﬁcient k-means on gpus. In Proceedings of

the 14th International Workshop on Data Management on New Hardware, pages 1–3, 2018.

[18] S. Milborrow, J. Morkel, and F. Nicolls. The MUCT Landmarked Face Database. Pattern

Recognition Association of South Africa, 2010. http://www.milbo.org/muct.

10

282

283

284

285

286

287

288

289

290

291

292

293

294

295

296

297

298

299

300

301

302

303

304

305

306

307

308

309

310

311

312

313

314

315

316

317

318

319

320

321

322

323

[19] A.Y. Ng, M.I. Jordan, and Y. Weiss. On spectral clustering: Analysis and an algorithm. In

Advances in neural information processing systems, pages 849–856, 2002.

[20] F. Nie, S. Pei, R. Wang, and X. Li. Fast clustering with co-clustering via discrete non-negative
matrix factorization for image identiﬁcation. In ICASSP 2020-2020 IEEE International Confer-
ence on Acoustics, Speech and Signal Processing (ICASSP), pages 2073–2077. IEEE, 2020.

[21] F. Nie, X. Wang, and H. Huang. Clustering and projected clustering with adaptive neighbors. In
Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and
Data Mining, KDD ’14, page 977–986, New York, NY, USA, 2014. Association for Computing
Machinery.

[22] J. Ortiz-Bejar, E.S. Tellez, M. Graff, J. Ortiz-Bejar, J.C. Jacobo, and A. Zamora-Mendez.
Performance analysis of k-means seeding algorithms. In 2019 IEEE International Autumn
Meeting on Power, Electronics and Computing (ROPEC), pages 1–6. IEEE, 2019.

[23] S. Pei, F. Nie, R. Wang, and X. Li. Efﬁcient clustering based on a uniﬁed view of k-means and

ratio-cut. Advances in Neural Information Processing Systems, 33, 2020.

[24] P.J. Phillips, H. Wechsler, J. Huang, and P.J. Rauss. The feret database and evaluation procedure

for face-recognition algorithms. Image and vision computing, 16(5):295–306, 1998.

[25] S.J. Phillips. Acceleration of k-means and related clustering algorithms. In Algorithm Engineer-

ing and Experiments, pages 166–177. Springer Berlin Heidelberg, 2002.

[26] S. Sarfraz, V. Sharma, and R. Stiefelhagen. Efﬁcient parameter-free clustering using ﬁrst
neighbor relations. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 8934–8943, 2019.

[27] S. Sengupta, J. Chen, C. Castillo, V.M. Patel, R. Chellappa, and D.W. Jacobs. Frontal to proﬁle
face veriﬁcation in the wild. In 2016 IEEE Winter Conference on Applications of Computer
Vision (WACV), pages 1–9. IEEE, 2016.

[28] S.A. Shah and V. Koltun. Robust continuous clustering. Proceedings of the National Academy

of Sciences, 114(37):9814–9819, 2017.

[29] J. Shi and J. Malik. Normalized cuts and image segmentation. IEEE Transactions on pattern

analysis and machine intelligence, 22(8):888–905, 2000.

[30] T. Sim, S. Baker, and M. Bsat. The cmu pose, illumination, and expression database. IEEE

Transactions on Pattern Analysis and Machine Intelligence, 25(12):1615–1618, 2003.

[31] M.E. Tipping and C.M. Bishop. Probabilistic principal component analysis. Journal of the

Royal Statistical Society: Series B (Statistical Methodology), 61(3):611–622, 1999.

[32] S. Xia, D. Peng, D. Meng, C. Zhang, G. Wang, E. Giem, W. Wei, and Z Chen. A fast adaptive
k-means with no bounds. IEEE Transactions on Pattern Analysis and Machine Intelligence,
pages 1–1, 2020.

[33] Y. Zhao, Y. Yuan, and Q. Wang. Fast spectral clustering for unsupervised hyperspectral image

classiﬁcation. Remote Sensing, 11(4):399, 2019.

[34] T. Zheng and W. Deng. Cross-pose lfw: A database for studying cross-pose face recognition
in unconstrained environments. Technical Report 18-01, Beijing University of Posts and
Telecommunications, February 2018.

[35] T. Zheng, W. Deng, and J. Hu. Cross-age LFW: A database for studying cross-age face

recognition in unconstrained environments. CoRR, abs/1708.08197, 2017.

11

324

325

326

327

328

329

330

331

332

333

334

335

336

337

338

339

340

341

342

343

344

345

346

347

348

349

350

351

352

353

354

355

356

357

358

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes]
(c) Did you discuss any potential negative societal impacts of your work? [N/A]
(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [N/A]
(b) Did you include complete proofs of all theoretical results? [N/A]

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-

mental results (either in the supplemental material or as a URL)? [Yes]

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)? [Yes]

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [Yes]

(d) Did you include the total amount of compute and the type of resources used (e.g., type

of GPUs, internal cluster, or cloud provider)? [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]
(b) Did you mention the license of the assets? [No]
(c) Did you include any new assets either in the supplemental material or as a URL? [No]
(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating? [Yes]

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable

information or offensive content? [No]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

12

