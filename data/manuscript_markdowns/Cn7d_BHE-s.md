Compressed Video Contrastive Learning

Yuqi Huo2,3,• Mingyu Ding4,• Haoyu Lu1,2 Nanyi Fei2,3
Zhiwu Lu1,2,∗ Ji-Rong Wen1,2 Ping Luo4
1Gaoling School of Artiﬁcial Intelligence, Renmin University of China, Beijing, China
2Beijing Key Laboratory of Big Data Management and Analysis Methods
3School of Information, Renmin University of China, Beijing, China
4The University of Hong Kong, Pokfulam, Hong Kong, China
{bnhony, luzhiwu}@ruc.edu.cn

• Equal contribution

∗ Corresponding author

Abstract

This work concerns self-supervised video representation learning (SSVRL), one
topic that has received much attention recently. Since videos are storage-intensive
and contain a rich source of visual content, models designed for SSVRL are
expected to be storage- and computation-efﬁcient, as well as effective. However,
most existing methods only focus on one of the two objectives, failing to consider
both at the same time. In this work, for the ﬁrst time, the seemingly contradictory
goals are simultaneously achieved by exploiting compressed videos and capturing
mutual information between two input streams. Speciﬁcally, a novel Motion Vector
based Cross Guidance Contrastive learning approach (MVCGC) is proposed. For
storage and computation efﬁciency, we choose to directly decode RGB frames
and motion vectors (that resemble low-resolution optical ﬂows) from compressed
videos on-the-ﬂy. To enhance the representation ability of the motion vectors, hence
the effectiveness of our method, we design a cross guidance contrastive learning
algorithm based on multi-instance InfoNCE loss, where motion vectors can take
supervision signals from RGB frames and vice versa. Comprehensive experiments
on two downstream tasks show that our MVCGC yields new state-of-the-art while
being signiﬁcantly more efﬁcient than its competitors.

1

Introduction

Recent self-supervised image representation learning approaches [He et al., 2020; Chen et al., 2020]
have been reported to outperform supervised ones on a wide range of downstream tasks by (1)
leveraging a large amount of unlabeled data available online for pre-training, and (2) designing a
powerful algorithm to discriminate unlabeled samples with different semantic meanings. However,
the situation for self-supervised video representation learning (SSVRL) is somewhat different:
self-supervised methods still perform worse than supervised ones, since videos are extremely storage-
intensive (scalability thus becomes a serious issue) and have a richer source of visual content (SSVRL
thus becomes very difﬁcult). Therefore, designing storage-and-computation-efﬁcient as well as
effective models for SSVRL remains a challenging problem that has not been well studied.

Existing state-of-the-art methods [Han et al., 2020b; Tao et al., 2020; Huo et al., 2021] mainly focus
on designing effective algorithms without considering the storage and computation costs during
large-scale video self-supervised training. Particularly, in order to leverage both appearance and
temporal information in the video, they exploit the optical ﬂow as an extra view to complement the
RGB stream. However, this procedure is both storage- and computation-intensive because of storing
the decoded frames and computing optical ﬂows, respectively (e.g., it costs more than 100GB for

35th Conference on Neural Information Processing Systems (NeurIPS 2021).

Figure 1: Processing time per-video (i.e., data pre-processing and inference) and the accuracies for
different methods on UCF101. Each method is represented by a circle, whose size represents the
storage size occupied by the test data. MVCGC achieves the highest accuracy under a signiﬁcantly
reduced storage budget compared to its counterparts. More details can be found in Table 1.

frame storage and a number of days for computing optical ﬂows [Han et al., 2020b] on the UCF101
dataset [Soomro et al., 2012]). This clearly hinders large-scale video self-supervised training.

Considering that videos are already stored in compressed formats to reduce storage requirements, a
natural choice is to resort to on-the-ﬂy decoding, i.e., decoding frames from compressed videos during
training/inference without introducing any extra storage. Moreover, since motion vectors encoded in
compressed videos resemble low-resolution optical ﬂows in describing local motions, they can be used
to avoid the costly optical ﬂow computation. Consequently, compressed videos inherently contain
both static and motion information that is suitable for video representation learning. Recently, a pretext
task based method called IMRNet [Yu et al., 2021] leverages the standard process in CoViAR [Wu et
al., 2018] to decode compressed videos, but inevitably encounters the inefﬁciency and ineffectiveness
problems: (1) The outdated CoViAR is not truly on-the-ﬂy, i.e., it re-encodes videos and stores
them before decoding, which is storage and computation inefﬁcient. (2) Motion vectors are less
discriminative and thus weaker than optical ﬂows, resulting in sub-optimal performance compared
with state-of-the-art methods using optical ﬂows, as shown in Figure 1. As a result, exploiting the
compressed videos efﬁciently, followed by learning discriminative representations, is still challenging.

Motivated by the above observations, we propose a Motion Vector based Cross Guidance Contrastive
learning approach (MVCGC), which has several appealing beneﬁts: (1) It is able to decode RGB
frames and motion vectors from compressed videos with various codecs. The storage and computation
budgets are alleviated since videos can be decoded without re-encoding. (2) Our MVCGC can learn
discriminative features from both RGB and motion vector streams. This is achieved by designing a
cross guidance contrastive learning algorithm based on multi-instance InfoNCE loss: each sample
pair is constructed by one RGB clip and one motion vector clip, and multiple positive samples can
be mined by calculating the similarity in two views. Therefore, the two views can take supervision
signals from each other to improve the representation quality of both streams, especially for the
motion vector stream (whose learned representations are even comparable to optical ﬂow features).

Our main contributions are three-fold: (1) We propose an efﬁcient and effective framework called
MVCGC that can learn representations from compressed videos directly. To the best of our knowledge,
we are the ﬁrst to exploit contrastive loss in compressed video self-supervised learning. (2) We
design a novel contrastive learning algorithm to capture mutual information between RGB frames
and motion vectors from compressed videos. Importantly, we even make the learned features of
motion vectors as representative as those of optical ﬂows. (3) Extensive experiments by applying the
learned video representations on two downstream tasks (i.e., action recognition and action retrieval)
across different benchmarks demonstrate that our MVCGC achieves state-of-the-art performance
while being more efﬁcient than its counterparts (see Figure 1).

2

10−1100101Processing Time Per-video (second)737679828588Top-1 Accuracy (%)MVCGC (ours)IMRNetCOCLRMemDPCIIC2 Related Work

Self-Supervised Video Representation Learning. Existing self-supervised methods for video
representation learning can be divided into two categories: (1) One common way to learn good
representations is to deﬁne and solve pretext tasks, such as ordering frames [Misra et al., 2016; Lee et
al., 2017; Xu et al., 2019; Luo et al., 2020], predicting the speed of videos [Yao et al., 2020; Benaim
et al., 2020; Wang et al., 2020], and solving space-time cubic puzzles [Kim et al., 2019]. (2) Another
more recent way focuses on instance discrimination, which leverages distance-based contrastive loss
and distinguishes the positive samples from a group of negative ones [Oord et al., 2018; He et al.,
2020; Chen et al., 2020]. Recently, many efforts based on contrastive learning have demonstrated
promising results. They either focus only on RGB frames [Han et al., 2019; Zhuang et al., 2020;
Wang et al., 2020; Kong et al., 2020; Wang et al., 2021] or introduce an additional optical ﬂow view
in order to achieve state-of-the-art accuracy [Han et al., 2020a; Tian et al., 2020; Tao et al., 2020;
Han et al., 2020b]. Considering the heavy computational and data-storage burdens for computing the
optical ﬂow, our proposed MVCGC replaces it by introducing the motion vector from the compressed
video as a new view, which signiﬁcantly cuts down the computational and storage costs. Moreover,
MVCGC is different from the previous approaches [Han et al., 2020a; Tian et al., 2020; Tao et al.,
2020; Han et al., 2020b] in how contrastive loss and sample pairs are deﬁned. CMC [Tian et al., 2020]
and IIC [Tao et al., 2020] use the single-instance InfoNCE loss, resulting in neglect of hard positives.
MemDPC [Han et al., 2020a] and CoCLR [Han et al., 2020b] construct sample pairs of contrastive
loss in RGB frames and optical ﬂows separately, ignoring the correspondence between different
views. In contrast, our MVCGC constructs sample pairs between different views of different clips
(i.e., RGB frames of one clip and motion vectors of another clip from the same video) and applies the
multi-instance InfoNCE loss. We use two views as cross guidance to train the representations of two
streams simultaneously and learn mutual information between RGB frames and motion vectors.

Compressed Video Representation Learning. Earlier approaches have explored the compressed
video for supervised recognition [Zhang et al., 2018; Wu et al., 2018; Shou et al., 2019], in which
large-scale supervised datasets (such as ImageNet [Russakovsky et al., 2015]) are used for pre-
training. In contrast, we focus on self-supervised learning, i.e., only unlabeled data is used in
the pre-training stage. After the ﬁrst exploration of using compressed video to learn “the arrow
of time”[Wei et al., 2018], a related work, IMRNet [Yu et al., 2021], also explores compressed
video self-supervised learning besides supervised learning. Our proposed MVCGC is fundamentally
different from IMRNet in three key aspects: (1) Instead of focusing on designing pretext tasks as in
IMRNet, our MVCGC leverages contrastive loss. (2) With the outdated MPEG-4 Part 2 codec [Huo
et al., 2020], IMRNet has to re-encode and store videos for data loading, whereas our MVCGC can
decode compressed videos on-the-ﬂy and save storage spaces. (3) Besides the decoded RGB and
motion vector streams, IMRNet leverages an additional residual stream computed from the two
streams, which has already been proved to only supply a marginal performance gain. In comparison,
our MVCGC utilizes the ﬁrst two streams and thus alleviates computation burdens.

3 Methodology

This work aims to learn robust video representations efﬁciently from compressed videos by designing
a contrastive learning algorithm. In this section, we ﬁrst review the basics of compressed video and
discuss the advantages of our decoding process compared to existing counterparts. Afterward, we
detail our MVCGC on how it learns effective feature representations and captures mutual information
between RGB frames and motion vectors from compressed videos.

3.1 Basics of Compressed Video

Considering the redundancy in consecutive frames, codecs are proposed to compress videos for
efﬁcient storage, i.e., compress one frame by reusing contents from another frame (termed reference
frame) and only store the change. According to the choice of reference frame, frames are categorized
into three types: I-frame (Intra-coded frame), P-frame (Predictive frame), and B-frame (Bi-predictive
frame). I-frame has no reference frame and is directly stored in image format, while P- and B-frame
take reference frames forwardly and bi-directionally, respectively, and store the change in the format
of the motion vector and residual: A codec ﬁrst divides a frame into macroblocks of size such

3

Figure 2: Usage of compressed videos (Comp. Video) for SSVRL: (a) Traditional methods decode
videos off-the-ﬂy; (b) IMRNet re-encodes videos, followed by the additional residuals computation
with RGB frames being discarded afterward; (c) MVCGC decodes RGB frames and motion vectors
on-the-ﬂy. “RGB", “MV", “OF", and “Res" denote RGB frames, motion vectors, optical ﬂows, and
residuals, respectively. Procedures colored with red are computation expensive, and their backgrounds,
which are colored with orange or blue, indicate the happening place of them, i.e., videos in the orange
background are stored on the disk while those in the blue are decoded in cache memory.

as 16x16 and then searches the most similar image patch in the reference frame for each of these
macroblocks. Lastly, the motion vector is represented by the displacements between macroblocks
and the searched image patches, and the residual is computed as the difference between the target
frame and its reference frame warped by the motion vector.

Discussion As shown in Figure 2, existing SSVRL methods can be categorized into three groups
according to how they exploit videos for SSVRL. (1) Traditional methods off-the-ﬂy decode all
RGB frames from compressed videos and compute optical ﬂows with RGB frames, which is storage-
and computation-intensive. (2) The latest IMRNet [Yu et al., 2021] decodes compressed videos by
leveraging the process in CoViAR [Wu et al., 2018], which only supports the outdated MPEG-4
Part 2 codec [Sikora, 1997] and lacks generalizability. In this way, videos with modern codecs
(e.g., H.264/AVC [Wiegand et al., 2003] and HEVC [Sullivan et al., 2012]) in current application
scenarios need to be re-encoded and re-stored, downgrading the computational efﬁciency. IMRNet
uses residuals of P-frames as a third view: since residuals can not be extracted in practice, it is
post-computed by two decoded views, i.e., RGB frames and motion vectors, which introduces extra
computation requirements. Moreover, in IMRNet, RGB frames are decoded only for computing
residuals, while being discarded afterward, hindering the data usage effectiveness. (3) Our proposed
MVCGC can decode RGB frames and motion vectors directly from various video codecs without the
aforementioned costly off-the-ﬂy frame extracting, video re-encoding, or residual computing, and is
thus high-efﬁciency and application-ﬂexibility.

3.2 Motion Vector based Cross Guidance Contrastive Learning

The goal of SSVRL is to train an encoder that can embed video samples effectively for various
downstream tasks (e.g., action recognition and retrieval). Towards this end, our method is based on
contrastive learning with InfoNCE loss [Oord et al., 2018], as shown in Figure 3. Next we start by
reviewing a base single-view model with InfoNCE loss, and then extend the base model using cross
guidance contrastive learning as our MVCGC.

Base Single-View Model with InfoNCE Loss Formally, let D = {x1, x2, · · · , xN } denotes a
dataset of N compressed video samples, where each clip xi (i = 1, 2, · · · , N ) consists of T frames.
In the single-view scenario, these clips are in the same view, i.e., D is a dataset of only RGB clips or
motion vectors. The objective of learning an effective encoding function f (·) for the single stream
is thus achieved by discriminating one positive sample of each video clip from its negatives with
InfoNCE loss [Oord et al., 2018].

Let t(·; θ) be an augmentation function applied to D, where θ is sampled from a set of transformations
Θ. For each clip xi ∈ D (i = 1, 2, · · · , N ), we denote the encoded feature vector of xi as zi = f (xi),
the positive sample set as Pi = {f (t(xi; θ))|θ ∼ Θ}, and the negative set as Ni = {f (t(xj; θ))|∀j (cid:54)=

4

DecodeRGBData Pre-processingInferenceRe-encodeDecode Compute (a)(b)(c)ComputeComp. VideoDiscardedDecode Comp. VideoComp. VideoComp. VideoOFResMVMVRGBRGBFigure 3: Overview of the proposed MVCGC architecture. (a) From a raw RGB clip and its
corresponding motion vector clip, MVCGC computes the anchor and the positive features, while
other samples in the dataset are used to construct the negative set. (b) The complete design of the
hard positive mining stage.

i, θ ∼ Θ}, where |Pi| = 1 and |Ni| = N − 1. The InfoNCE loss is then deﬁned as:

LInfoNCE =

(cid:88)

− log

exp (zi · zp/τ ) + (cid:80)
where zp ∈ Pi is the only positive sample of xi, zi · zp (or zi · zn) refers to the dot product between
two vectors, and τ > 0 is a temperature parameter. This instance discrimination loss forces the base
model to learn a higher similarity score between a given sample zi and its augmented/target sample
zp while a lower one between zi and other instances zn ∈ Ni.

exp (zi · zn/τ )

zn∈Ni

i

,

(1)

exp (zi · zp/τ )

Cross Guidance Contrastive Learning with Multiple Views While the InfoNCE loss can learn
discriminative representations in a single view well, it faces two key challenges when being extended
to multiple views: (1) For multiple sources of information (i.e., RGB frames, motion vectors, and
optical ﬂows), simply applying an InfoNCE loss for each stream ignores the corresponding mutual
information across these streams. (2) It aims only to discriminate the augmented sample from all
other instances, with the assumption that the target sample is semantically different from all the others.
This is counter-intuitive, since different video clips may also have similar semantic content, i.e., hard
positives are neglected by InfoNCE. We thus propose MVCGC to address these two problems.
Concretely, MVCGC decodes an RGB stream (RGB) ri ∈ RT ×H×W ×3 and a motion vector
stream (MV) mi ∈ RT ×H×W ×2 as two input views for each video clip xi ∈ D (i = 1, 2, · · · , N ).
MVCGC aims to learn two encoding functions f (r)(·), f (m)(·) for the two streams, respectively. To
model the interaction between two streams, we extend the single-view InfoNCE loss in Eq. (1) by
choosing positive and negative samples from the other view for each stream. Speciﬁcally, given the
query of an RGB clip z(r)
, the objective is to compare its similarity among the features of motion
vector clips, i.e., emitting higher similarity between the positive sample z(m)
, p ∈ Pi than with those
p
of other negative instances z(m)
and P (m)
) of each ri or mi are
i
also expanded for the hard positive mining by adding the top-k similar clips:

n , n ∈ Ni. The positive sets (i.e., P (r)

i

i

P (r)
i = {f (m)(t(mj; θ))|j ∈ Si, θ ∼ Θ},
P (m)
i = {f (r) (t( rj; θ))|j ∈ Si, θ ∼ Θ},
· z(r)
j

) ∩ top-k(z(m)

· z(m)
j

i

(2)

(3)

where Si = {j|j ∈ (top-k(z(r)
set of positive samples, z(r)
the features in the two views as the cross guidance of each other:

i = f (r)(ri) and z(m)

)), j = (1, 2, · · · , N )} denotes the index
i = f (m)(mi). Our learning objective is thus to use

i

Li

MVCGC = − log

(cid:80)

z(m)
p

− log

(cid:80)

z(r)
p

exp(z(r)

i
(cid:80)

exp(z(m)

i

(cid:80)

exp(z(r)
i
/τ ) + (cid:80)

z(m)
p
·z(m)
p

exp(z(m)
z(r)
p
p /τ ) + (cid:80)
·z(r)

i

·z(m)
p

/τ )

z(m)
n
·z(r)

p /τ )

exp(z(m)

·z(r)

n /τ )

i

z(r)
n

exp(z(r)

·z(m)

n /τ )

i

(4)

,

5

LMVCGCCross GuidanceRGBMVf(m)( )Hard Positive MiningPNPPN…∩PNNPP…PNNPN…PNNPN…01234…PNNPN…PNNPN…PNNNN…PNNNN…Top-k SimilarityADDTop-k SimilarityADD…………（a）（b）Cross Guidancef(r)( )SiPiNit( ; θ）t( ; θ）PiNizi(r)zi(m)where z(m)
p ∈ P (r)
all training samples:

i

, z(m)

n ∈ N (r)

i

, z(r)

p ∈ P (m)

i

, and z(r)

n ∈ N (m)

i

LMVCGC =

Li

MVCGC.

(cid:88)

i

. The ﬁnal loss is computed over

(5)

Discussion Note that our MVCGC is different from other methods with discriminative InfoNCE
loss in two aspects. First, differing from CMC [Tian et al., 2020] and IIC [Tao et al., 2020] which
only mine a single positive and regard the hard positives as negative samples, MVCGC incorporates
the learning from hard positives. Second, the co-training in CoCLR [Han et al., 2020b] fails to learn
the correspondence between two views (i.e., there is no information exchange in the pre-learning
stage), but our MVCGC leverages cross guidance to capture mutual information between two streams.
We provide the algorithm in the supplementary material.

3.3 MVCGC Algorithm

The detailed procedure of our MVCGC is summarized in Algorithm 1.

Algorithm 1 Motion Vector based Cross Guidance Contrastive Learning (MVCGC)
Require: Compressed video dataset D.

An RGB clip ri ∈ RT ×H×W ×3 and a motion vector clip mi ∈ RT ×H×W ×2 for xi ∈ D
(i = 1, 2, · · · , N ).
Encoders f (r)(·), f (m)(·) for RGB and motion vector streams, respectively.
The augmentation set t(·; θ), θ ∼ Θ.
Temperature τ , top-k hard positive mining parameter k.

, ∀i ∈ [1, N ], j (cid:54)= i;

, z(r)
j

, z(m)
i

, z(m)
j

1: for epoch = 1 to #epochs: do
2:
3:

Compute z(r)
i
Compute Si;
Obtain P (r)
Compute cross-entropy loss LMVCGC;
Update model parameters;

4:
5:
6:
7: end for
8: return Optimized encoders f (r)(·) and f (m)(·).

, N (m)
i

, P (m)
i

, N (r)
i

;

i

4 Experiments

4.1 Datasets

In this paper, we use UCF101 [Soomro et al., 2012] and Kinetics-400 (K400) [Kay et al., 2017] for
self-supervised pre-training. UCF101 contains 13,320 videos with 101 action classes and has three
standard training/test splits. While K400 is a larger dataset consisting of 400 human action classes
and has 230k/20k clips for training/validation, respectively. In the pre-training stage, compressed
videos in the ﬁrst training set of UCF101 and the training split of K400 are used without labels.
Following the common practice [Han et al., 2020b], we benchmark downstream evaluation tasks on
the ﬁrst test set of UCF101, and the test split 1 of HMDB51 [Kuehne et al., 2011], a relatively small
action dataset containing 6,766 videos with 51 categories.

4.2

Implementation details

We follow the state-of-the-art method [Han et al., 2020b] for adopting S3D [Xie et al., 2018]
architecture as the backbone feature extractor for all experiments and apply the momentum-updated
history queue as in MoCo [He et al., 2020] for the framework of contrastive learning to leverage
a larger number of negative samples. Datasets used contain videos with various codecs, which are
directly decoded into RGB frames and motion vectors. We implement MVCGC based on “pyav”,
a Pythonic binding for the FFmpeg libraries used for decoding RGB frames and motion vectors
on-the-ﬂy. A non-linear projection head is attached above each encoder during the pre-training stage
and is removed for downstream evaluations.

6

Table 1: Results of processing speeds, storages and accuracies on UCF101. The underline represents
the second-best result.

Method

IIC

MemDPC

CoCLR

IMRNet

[Tao et al., 2020] [Han et al., 2020a] [Han et al., 2020b] [Yu et al., 2021]

MVCGC
(ours)

Pre-processing Time (s) ↓
Inference Time (ms) ↓
Total Time (s) ↓
Storage (GB)↓
Top-1 Acc (%)↑

12.7
69.3
12.8
38.0
72.7

12.7
253.1
13.0
38.0
84.3

13.2
248.7
13.4
35.4
87.3

0.128
227.2
0.355
6.1
76.8

0.008
70.3
0.078
1.9
87.4

Our MVCGC proceeds in two stages: the initialization stage and the cross guidance stage. In the ﬁrst
stage, the encoder of each view (i.e., RGB or MV) is initialized with Eq. (1) independently. After
initialization, these two encoders are cross-trained together to minimize the loss in Eq. (5). To cache a
large number of features, we adopt a momentum-updated history queue as in MoCo [He et al., 2020],
which is used in both two pre-training stages. For the pre-training on UCF101, temperature τ = 0.07,
momentum m = 0.999 and queue size 2048 are used, while queue size is set to 16384 on K400.
When pre-training on UCF101, the initialization stage lasts 300 epochs for each stream, and we then
continually train the cross guidance for another 200 epochs. On K400, we train 200 epochs for each
stream in the initialization stage and 50 epochs for cross guidance contrastive learning. 100 and 500
epochs are used for linear and fully ﬁne-tuning, respectively. We use the Adam optimizer with a 1e-4
learning rate and 1e-5 weight decay for pre-training and the SGD optimizer with a 1e-1 learning rate
and 1e-3 weight decay for ﬁne-tuning. The learning rate is decayed down by 1/10 twice when the
validation loss plateaus. The hyper-parameter k in MVCGC is set as 5 according to the ablation study.
All experiments are trained on 4 TiTan RTX GPUs, with a batch size of 32 samples per GPU.

We make evaluation on two downstream tasks: action classiﬁcation and video retrieval. In the
action classiﬁcation task, two evaluation settings are used: (1) linear probing, where parameters
of the learned encoder are frozen and only a single linear layer is ﬁne-tuned; (2) fully ﬁne-tuning,
where the encoder is ﬁne-tuned together. At the inference stage in either of these two settings, we
follow the same procedure as in the previous work [Han et al., 2020b]: for each video, we spatially
apply ten-crops and temporally take clips with moving windows, and then average the predicted
probabilities. In the video retrieval task, we adopt widely-used evaluation metrics [Han et al., 2020b;
Tao et al., 2020], i.e., we leverage the extracted features for nearest-neighbour (NN) retrieval without
ﬁne-tuning. More details can be found in the supplementary material.

4.3 Efﬁciency Results

Table 1 presents the efﬁciency results on UCF101 compared with four state-of-the-art self-supervised
methods: IIC [Tao et al., 2020], MemDPC [Han et al., 2020a], CoCLR [Han et al., 2020b], and
IMRNet [Yu et al., 2021]. We use per-video processing time, storage spaces, and top-1 accuracies as
evaluation metrics under the fully ﬁne-tuning setting. All methods are measured in exactly the same
environment: Intel Xeon 5118 CPUs and a Titan RTX GPU. The total processing time consists of
two parts: data pre-processing and inference time. In the pre-processing stage, IIC, MemDPC, and
CoCLR decode RGB frames and calculate optical ﬂows with the TV-L1 algorithm, while IMRNet
re-encodes the raw compressed videos. We can observe that: (1) our MVCGC has the fastest speed
and reduces the storage budgets by 95% and 70% compared with traditional methods and IMRNet,
respectively. (2) For inference, the data is loaded from disks and is used for network forwarding.
Although the time costs may differ because of the backbone architectures used by different methods,
MVCGC still gets a comparable result to IIC and is faster than other three approaches. Nevertheless,
the majority of the total process is dominated by data pre-processing, where MVCGC signiﬁcantly
outperforms competitors. (3) Among fully ﬁne-tuning methods, MVCGC outperforms those using
either residuals or optical ﬂows, showing the effectiveness of our proposed algorithm.

4.4 Ablation Study

We ﬁrst show how our proposed MVCGC beneﬁts the learned representations of motion vectors in
Table 2, where the top-1 accuracies of action classiﬁcation and retrieval performance are used as
evaluation metrics since their evaluation is fast. We can make the following four observations: (1) The
upper part of the table shows the results of different views after initialization, where there is clearly a

7

Table 2: Evaluation of motion vectors (MV) and optical ﬂows (OF) on downstream action classiﬁca-
tion and retrieval on UCF101.

Method

Pre-training

Evaluation

Linear Probe

Retrieval

Init.
Init.
Init.
Init.

MVCGC
MVCGC

MV
OF
OFLow
MVCoViAR
RGB+OF
RGB+MV

MV
OF
OFLow
MVCoViAR
OF
MV

65.0
66.8
51.3
63.4

39.9
45.2
30.4
39.2

73.1 (+6.3)
73.8 (+8.8)

60.6 (+15.4)
60.8 (+20.9)

Figure 4: Visualization of the last feature maps from the models linear probed on UCF101. Each
row shows the attention visualization of motion vectors before and after MVCGC pre-training and
the ﬁnal result of optical ﬂows, respectively. Attention map is generated with 32-frames clip inputs
and applied to the middle frame in the video clips.

performance gap between the motion vector and the optical ﬂow. To verify whether this gap originates
from the lower resolution of the motion vector (i.e., same value in each macroblock), we also compare
with the lower resolution of the optical ﬂow (i.e., OFLow). We ﬁnd that by only reducing the resolution
of the raw optical ﬂow to a half (i.e., same value in each 2 × 2 macroblock), it performs even worse
than the motion vector. This comparison demonstrates that the motion vector, although having lower
resolutions than the optical ﬂow, is also able to well represent the action motions. (2) Since motion
vectors may appear differently based on codec settings (i.e., different codecs and coding qualities),
we also conduct experiment with the motion vectors extracted by the previous procedure [Yu et al.,
2021; Wu et al., 2018] (i.e., MVCoViAR), where they are all decoded from P-frames. We ﬁnd that
our MVCGC is robust to motion vectors under different encoding conﬁgurations. (3) The motion
vector view (or the optical ﬂow view) can signiﬁcantly beneﬁt from the RGB view by our MVCGC,
validating the effectiveness of our proposed cross guidance pre-training used in our MVCGC. (4)
The performance of our MVCGC using motion vector + RGB is comparable to (and even slightly
better than) that of our MVCGC using optical ﬂow + RGB. This ﬁnding is also supported by the
visualization results shown in Figure 4.

We then demonstrate the advantages of our proposed MVCGC over the closely-related method
CoCLR [Han et al., 2020b] and also our choices for the hyper-parameters of MVCGC in Table 3,
where we focus on the performance of the RGB network during evaluation. First, from the upper part
of Table 3, we ﬁnd that our MVCGC indeed helps the RGB network to learn more discriminative
representations than CoCLR, when the extra information of the motion vector view (or the optical ﬂow
view) is leveraged. This shows the effectiveness of our cross guidance contrastive learning. Second,
according to the lower part of Table 3, we select k = 5 as the optimal setting among k = 1/5/10/20.
Particularly, MVCGC with ‘w/o init.’ means that there is no initialization stage before the cross
guidance stage, which performs the worse. Moreover, MVCGC with ‘union’ means that the index set
Si is generated by the union of positive candidates in two streams, which performs worse than the
intersection used in our MVCGC.

8

ApplyingLipstickShootingArrowBikingOnBalancingBeamMVinit.MVMVCGCOFMVCGCMVinit.MVMVCGCOFMVCGCTable 3: Evaluation of different pre-training settings (i.e., algorithms and hyper-parameters) on
downstream action classiﬁcation and video retrieval on UCF101. All results are evaluated using only
the RGB view. The underline represents the second-best result.

Method

Init.
CoCLR
CoCLR
MVCGC
MVCGC(k=5)

MVCGC(k=5, w/o init.)
MVCGC(k=1)
MVCGC(k=10)
MVCGC(k=20)
MVCGC(k=5, union)

Pre-training

Evaluation

Linear Probe

Retrieval

RGB
RGB+OF
RGB+MV
RGB+OF
RGB+MV

RGB+MV
RGB+MV
RGB+MV
RGB+MV
RGB+MV

RGB
RGB
RGB
RGB
RGB

RGB
RGB
RGB
RGB
RGB

52.3
70.2(+17.9)
67.4(+15.1)
70.6(+18.3)
73.1(+20.8)

33.1
51.8(+18.7)
53.8(+20.7)
58.6(+25.5)
60.8(+27.7)

65.0
71.5
72.7
71.5
71.3

48.4
60.0
61.4
59.7
57.7

Table 4: The full table of comparison with state-of-the-art approaches on linear probing (Lin.) and
fully ﬁne-tuning (Full.) on UCF101 and HMDB51 benchmarks. Top-1 accuracies are reported.
“Dataset” denotes the pre-training dataset and “Duration” is the total length of videos. Methods
marked with ∗ use multiple views for pre-training. ‘Test View(s)’ denotes views used in evaluation,
i.e., RGB, motion vector(MV), optical ﬂow(OF), residual(Res), Audio, and Text.
Method

Dataset(Duration) UCF UCF HMDB HMDB

Test View(s)

Arch.

Res.

VCP [Luo et al., 2020]
PRP [Yao et al., 2020]
Pace [Wang et al., 2020]
DSM∗ [Wang et al., 2021]
CMC∗ [Tian et al., 2020]
CoCLR∗ [Han et al., 2020b]
MVCGC (ours)∗
IIC∗ [Tao et al., 2020]
MemDPC∗ [Han et al., 2020a]
CoCLR∗ [Han et al., 2020b]
MVCGC (ours)∗

RGB
RGB
RGB
RGB
RGB
RGB
RGB

RGB+OF
RGB+OF
RGB+OF
RGB+MV

C3D

16*112
R(2+1)D-18 16*112
R(2+1)D-18 16*112
16*112
25*256
32*128
32*128

C3D
CaffeNet
S3D
S3D

R3D-18

16*112
R2D3D-34 40*128
32*128
32*128

S3D
S3D

UCF(1d)
UCF(1d)
UCF(1d)
UCF(1d)
UCF(1d)
UCF(1d)
UCF(1d)

UCF(1d)
UCF(1d)
UCF(1d)
UCF(1d)

S3D-G
R3D-34

RGB
RGB
RGB
RGB
RGB
RGB
RGB
RGB

K400(28d)
K400(28d)
K400(28d)
K400(28d)
K400(28d)
K400(28d)
K400(28d)
K400(28d)

R(2+1)D-18 8*112
R2D3D-34 40*224
R(2+1)D-18 16*112
16*224
16*224
SlowFast-18 16*112
32*128
32*128

CCL [Kong et al., 2020]
DPC [Han et al., 2019]
Pace [Wang et al., 2020]
SpeedNet [Benaim et al., 2020]
DSM∗ [Wang et al., 2021]
VIE [Zhuang et al., 2020]
CoCLR∗ [Han et al., 2020b]
MVCGC (ours)∗
AVTS∗ [Korbar et al., 2018]
ELO∗ [Piergiovanni et al., 2020] RGB+Audio+OF R(2+1)D-50 32*224 YouTube800M(1.9y)
XDC∗ [Alwassel et al., 2020]
GDT∗ [Patrick et al., 2020]
CBT∗ [Sun et al., 2019]
MIL-NCE∗ [Miech et al., 2020]
IMRNet∗ [Yu et al., 2021]
MemDPC∗ [Han et al., 2020a]
CoCLR∗ [Han et al., 2020b]
MVCGC (ours)∗

R(2+1)D-18 32*224
R(2+1)D-18 32*224
16*112
32*200 Howto100M(15y)

30*224
R2D3D-34 40*224
32*128
32*128

RGB+MV+Res
RGB+OF
RGB+OF
RGB+MV

RGB+Audio
RGB+Audio
RGB+Text
RGB+Text

K400(28d)
K400(28d)
K400(28d)
K400(28d)

K400(28d)
K400(28d)
K400(28d)

R(2+1)D-18 25*224

RGB+Audio

K400(28d)

S3D
S3D

S3D
S3D

S3D
S3D

R3D-50

Lin. Full.

–
–
–
–
–

68.5
72.1
75.9
70.3
59.1
70.2 81.4
73.1 82.0

–
–

72.7
84.3
72.1 87.3
77.2 87.4

–
–
–
–
–

52.1 69.4
75.7
77.1
81.1
78.2
80.4
74.5 87.9
75.4 88.3

–
–
–
–

86.2
93.8
86.8
89.3
54.0 79.5
82.7 91.3

–

77.3
54.1 86.1
77.8 90.6
78.0 90.8

Lin.

–
–
–
–
–
39.1
41.1

–
–
40.2
41.0

27.8
–
–
–
–
–
46.1
49.7

–
64.5
–
–
29.5
53.1

–
30.5
52.4
53.0

Full.

32.5
35.0
35.9
40.5
26.7
52.1
58.4

36.8
–
58.7
59.7

37.8
35.7
36.6
48.8
52.8
52.5
54.6
61.4

52.3
67.4
52.6
60.0
44.6
61.0

47.5
54.5
62.9
63.4

4.5 Comparison with State-of-the-Art

In this section, we compare our MVCGC with the state-of-the-art self-supervised SSVRL approaches
on the action classiﬁcation and video retrieval tasks. For the action classiﬁcation task, we provide
the comparative results in Table 4, by pre-training on UCF101 and Kinetics-400 and then evaluating
with either linear probing or fully ﬁne-tuning on UCF101 and HMDB51. We list recent approaches
evaluated on the same benchmark, trying to compare with them as fairly as we can, although variations
are unavoidable in terms of architecture, training data, and resolution. The ‘Test View(s)’ column
refers to the view(s) used for test/evaluation, e.g., the result of MVCGC in ‘R+MV’ views is obtained
by averaging the predictions from RGB and motion vector networks. We can observe that: (1)

9

Table 5: Comparison with state-of-the-art methods in nearest-neighbour video retrieval on UCF101
and HMDB51. Videos in the test set are used to retrieve the videos in the training set, and R@k
is reported with k ∈ {1, 5, 10, 20, 50}. All compared methods are pre-trained on UCF101 except
SpeedNet pre-trained on larger Kinetics-400. Methods marked with ∗ use audio for pre-training.

Method

Retrieval with RGB only:
VCP [Luo et al., 2020]
SpeedNet [Benaim et al., 2020]
DSM [Wang et al., 2021]
PRP [Yao et al., 2020]
MemDPC [Han et al., 2020a]
Pace [Wang et al., 2020]
CCL [Kong et al., 2020]
CoCLR [Han et al., 2020b]
GDT∗ [Patrick et al., 2020]
MVCGC (ours)

Retrieval with two views:
IIC [Tao et al., 2020]
CoCLR [Han et al., 2020b]
MVCGC (ours)

R@1 R@5 R@10 R@20 R@50 R@1 R@5 R@10 R@20 R@50

UCF

HMDB

18.6
13.0
16.8
23.2
20.2
31.9
32.7
53.3
57.4
60.8

42.4
55.9
66.1

33.6
28.1
33.4
38.1
40.4
49.7
42.5
69.4
73.4
74.1

60.9
70.8
79.1

42.5
37.5
43.4
46.0
52.4
59.2
50.8
76.6
80.8
79.8

69.2
76.9
84.0

53.5
49.5
54.6
55.7
64.7
68.9
61.2
82.0
88.1
85.8

77.1
82.5
89.0

68.1
65.0
70.7
68.4
–
80.2
68.9
–
92.9
92.6

86.5
–
93.7

7.6
–
8.2
10.5
7.7
12.5
–
23.2
25.4
24.1

19.7
26.1
25.6

24.4
–
25.9
27.2
25.7
32.2
–
43.2
51.4
49.7

42.9
45.8
49.2

36.3
–
38.1
40.4
40.6
45.4
–
53.5
63.9
61.3

57.1
57.9
60.7

53.6
–
52.0
56.2
57.7
61.0
–
65.5
75.0
73.3

70.6
69.7
73.3

76.4
–
75.0
75.9
–
80.7
–
–
87.8
87.5

85.9
–
86.5

Our MVCGC outperforms most of latest methods under both linear probing and fully ﬁne-tuning
settings on both benchmarks. (2) Our MVCGC using motion vectors from compressed videos
clearly beats CMC [Tian et al., 2020] and CoCLR [Han et al., 2020b] which exploit optical ﬂows
for pre-training. (3) Compared to IMRNet [Yu et al., 2021] that also utilizes compressed videos
for self-supervised learning, our method achieves signiﬁcantly better results, demonstrating both
efﬁciency and effectiveness of MVCGC. (4) Our MVCGC outperforms its competitors under the
pre-training on Kinetics-400 and ﬁne-tuning on UCF101/HMDB51 setting, indicating the robustness
of MVCGC to process motion vectors that are obtained with different codecs. (5) It can be observed
that we outperform some approaches that exploit the correspondence of visual information with text
or audio. This is impressive since we only focus on visual representations and only use visual-related
views (i.e., RGB, motion vectors, optical ﬂows).

In Table 5, we evaluate our MVCGC in video retrieval on both UCF101 and HMDB51 We can see
that our MVCGC signiﬁcantly outperforms all compared methods in terms of all ﬁve metrics on
UCF101 and four of the ﬁve metrics on HMDB51, achieving new state-of-the-art. Particularly, our
obtained retrieval results by retrieval with only motion vectors are even higher than those of the
latest competitor CoCLR [Han et al., 2020b] which uses optical ﬂows for retrieval, indicating the
effectiveness of our MVCGC. Furthermore, we obtain comparable results with methods using extra
audio modalities [Patrick et al., 2020], indicating the effectiveness of MVCGC.

5 Conclusion

We have proposed a novel self-supervised video representation learning method named Motion Vector
based Cross Guidance Contrastive Learning (MVCGC). By introducing on-the-ﬂy decoded motion
vectors and cross guidance contrastive learning, our proposed MVCGC has for the ﬁrst time leveraged
contrastive loss in compressed video self-supervised learning. Extensive experiments are carried
out to validate the efﬁciency and effectiveness of MVCGC. Importantly, our approach achieves new
state-of-the-art on two downstream tasks across different benchmarks.

Acknowledgments and Disclosure of Funding

This work was supported in part by National Natural Science Foundation of China (61976220 and
61832017), Beijing Outstanding Young Scientist Program (BJJWZYJH012019100020098), and Open
Project Program Foundation of Key Laboratory of Opto-Electronics Information Processing, Chinese
Academy of Sciences (OEIP-O-202006). Ping Luo was supported by the General Research Fund of
Hong Kong No.27208720.

10

References

Humam Alwassel, Dhruv Mahajan, Lorenzo Torresani, Bernard Ghanem, and Du Tran. Self-supervised learning

by cross-modal audio-video clustering. In NeurIPS, 2020.

Sagie Benaim, Ariel Ephrat, Oran Lang, Inbar Mosseri, William T. Freeman, Michael Rubinstein, Michal Irani,

and Tali Dekel. Speednet: Learning the speediness in videos. In CVPR, pages 9922–9931, 2020.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive

learning of visual representations. In ICML, pages 10709–10719, 2020.

Tengda Han, Weidi Xie, and Andrew Zisserman. Video representation learning by dense predictive coding. In

ICCVW, pages 1483–1492, 2019.

Tengda Han, Weidi Xie, and Andrew Zisserman. Memory-augmented dense predictive coding for video

representation learning. In ECCV, pages 312–329, 2020.

Tengda Han, Weidi Xie, and Andrew Zisserman. Self-supervised co-training for video representation learning.

In NeurIPS, 2020.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised

visual representation learning. In CVPR, pages 9729–9738, 2020.

Yuqi Huo, Xiaoli Xu, Yao Lu, Yulei Niu, Mingyu Ding, Zhiwu Lu, Tao Xiang, and Ji-Rong Wen. Lightweight

action recognition in compressed videos. In ECCVW, 2020.

Yuqi Huo, Mingyu Ding, Haoyu Lu, Ziyuan Huang, Mingqian Tang, Zhiwu Lu, and Tao Xiang. Self-supervised

video representation learning with constrained spatiotemporal jigsaw. In IJCAI, 2021.

Will Kay, João Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio
Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human
action video dataset. arXiv preprint arXiv:1705.06950, 2017.

Dahun Kim, Donghyeon Cho, and In So Kweon. Self-supervised video representation learning with space-time

cubic puzzles. In AAAI, pages 8545–8552, 2019.

Quan Kong, Wenpeng Wei, Ziwei Deng, Tomoaki Yoshinaga, and Tomokazu Murakami. Cycle-contrast for

self-supervised video representation learning. In NeurIPS, 2020.

Bruno Korbar, Du Tran, and Lorenzo Torresani. Cooperative learning of audio and video models from self-

supervised synchronization. In NeurIPS, pages 7763–7774, 2018.

Hildegard Kuehne, Hueihan Jhuang, Estibaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: A large

video database for human motion recognition. In ICCV, pages 2556–2563, 2011.

Hsin-Ying Lee, Jia-Bin Huang, Maneesh Singh, and Ming-Hsuan Yang. Unsupervised representation learning

by sorting sequences. In ICCV, pages 667–676, 2017.

Dezhao Luo, Chang Liu, Yu Zhou, Dongbao Yang, Can Ma, Qixiang Ye, and Weiping Wang. Video cloze

procedure for self-supervised spatio-temporal learning. In AAAI, 2020.

Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-
to-end learning of visual representations from uncurated instructional videos. In CVPR, pages 9879–9889,
2020.

Ishan Misra, C. Lawrence Zitnick, and Martial Hebert. Shufﬂe and learn: Unsupervised learning using temporal

order veriﬁcation. In ECCV, pages 527–544, 2016.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding.

arXiv preprint arXiv:1807.03748, 2018.

Mandela Patrick, Yuki M. Asano, Ruth Fong, João F. Henriques, Geoffrey Zweig, and Andrea Vedaldi. Multi-
modal self-supervision from generalized data transformations. arXiv preprint arXiv:2003.04298, 2020.

AJ Piergiovanni, Anelia Angelova, and Michael S. Ryoo. Evolving losses for unsupervised video representation

learning. In CVPR, pages 133–142, 2020.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej
Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. IJCV,
115(3):211–252, 2015.

11

Zheng Shou, Xudong Lin, Yannis Kalantidis, Laura Sevilla-Lara, Marcus Rohrbach, Shih-Fu Chang, and
Zhicheng Yan. Dmc-net: Generating discriminative motion cues for fast compressed video action recognition.
In CVPR, pages 1268–1277, 2019.

Thomas Sikora. The mpeg-4 video standard veriﬁcation model. IEEE TCSVT, 7(1):19–31, 1997.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes

from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

Gary J. Sullivan, Jens-Rainer Ohm, Woo-Jin Han, and Thomas Wiegand. Overview of the high efﬁciency video

coding (hevc) standard. IEEE TCSVT, 22(12):1649–1668, 2012.

Chen Sun, Fabien Baradel, Kevin Murphy, and Cordelia Schmid. Learning video representations using contrastive

bidirectional transformer. arXiv preprint arXiv:1906.05743, 2019.

Li Tao, Xueting Wang, and Toshihiko Yamasaki. Self-supervised video representation learning using inter-intra

contrastive framework. In ACMMM, pages 2193–2201, 2020.

Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In ECCV, pages 776–794,

2020.

Jiangliu Wang, Jianbo Jiao, and Yun-Hui Liu. Self-supervised video representation learning by pace prediction.

In ECCV, pages 504–521, 2020.

Jinpeng Wang, Yuting Gao, Ke Li, Xinyang Jiang, Xiaowei Guo, Rongrong Ji, and Xing Sun. Enhancing

unsupervised video representation learning by decoupling the scene and the motion. In AAAI, 2021.

Donglai Wei, Joseph Lim, Andrew Zisserman, and William T. Freeman. Learning and using the arrow of time.

In CVPR, pages 8052–8060, 2018.

Thomas Wiegand, Gary J. Sullivan, Gisle Bjontegaard, and Ajay Luthra. Overview of the h. 264/avc video

coding standard. IEEE TCSVT, 13(7):560–576, 2003.

Chao-Yuan Wu, Manzil Zaheer, Hexiang Hu, R. Manmatha, Alexander J. Smola, and Philipp Krähenbühl.

Compressed video action recognition. In CVPR, pages 6026–6035, 2018.

Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature

learning: Speed-accuracy trade-offs in video classiﬁcation. In ECCV, pages 318–335, 2018.

Dejing Xu, Jun Xiao, Zhou Zhao, Jian Shao, Di Xie, and Yueting Zhuang. Self-supervised spatiotemporal

learning via video clip order prediction. In CVPR, pages 10334–10343, 2019.

Yuan Yao, Chang Liu, Dezhao Luo, Yu Zhou, and Qixiang Ye. Video playback rate perception for self-supervised

spatio-temporal representation learning. In CVPR, pages 6548–6557, 2020.

Youngjae Yu, Sangho Lee, Gunhee Kim, and Yale Song. Self-supervised learning of compressed video

representations. In ICLR, 2021.

Bowen Zhang, Limin Wang, Zhe Wang, Yu Qiao, and Hanli Wang. Real-time action recognition with deeply

transferred motion vector cnns. TIP, 27(5):2326–2339, 2018.

Chengxu Zhuang, Tianwei She, Alex Andonian, Max Sobol Mark, and Daniel Yamins. Unsupervised learning

from video with deep neural embeddings. In CVPR, pages 9563–9572, 2020.

12

