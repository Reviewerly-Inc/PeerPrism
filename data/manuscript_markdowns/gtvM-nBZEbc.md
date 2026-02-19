Under review as a conference paper at ICLR 2022

LEARNING VISUAL-LINGUISTIC ADEQUACY, FIDELITY,
AND FLUENCY FOR NOVEL OBJECT CAPTIONING

Anonymous authors
Paper under double-blind review

ABSTRACT

Novel object captioning (NOC) learns image captioning models for describing
objects or visual concepts which are unseen (i.e., novel) in the training captions.
Such captioning models need to sufﬁciently describe such visual data with ﬂuent
and natural language expression. In other words, we expect the produced captions
being linguistically ﬂuent, containing novel objects of interest, and ﬁtting the vi-
sual concept of the image. The above three aspects thus correspond to ﬂuency,
ﬁdelity, and adequacy, respectively. However, most novel object captioning mod-
els are not explicitly designed to address the aforementioned properties due to the
absence of caption annotations. In this paper, we start by providing an insight
into the relationship between the above properties and existing visual/language
models. Then, we present VLAF2, for learning Visual-Linguistic Adequacy, Fi-
delity, and Fluency, which utilizes linguistics observed from captions for describ-
ing visual information of images with novel objects. More speciﬁcally, we revisit
BERT and CLIP, and explain how we leverage the intrinsic language knowledge
from such popular models to reward captions with precise and rich visual con-
tent associated with novel images. To validate the effectiveness of our framework,
we conduct extensive experiments on the nocaps dataset. Our method not only
performs favorably against state-of-the-art novel captioning models in all caption
evaluation metrics, but also surpasses the SPICE scores of human baseline. We
perform quantitative and qualitative analysis to demonstrate how our model gen-
erates novel object captions with improved ﬂuency, ﬁdelity, and adequacy. Imple-
mentation details and code are available in the supplementary materials.

1

INTRODUCTION

Novel Object Captioning (NOC) (Agrawal et al., 2019) requires captioning models to accurately
capture images containing novel objects unseen during training captions, and to describe such data
with ﬂuent and grammatically correct sentences. Despite impressive benchmark performance on
COCO Captions (Chen et al., 2015) and Flickr (Young et al., 2014), existing image captioning
models (Gao et al., 2019; Huang et al., 2019; Wang et al., 2019; Guo et al., 2020; Pan et al., 2020;
Cornia et al., 2020; Zhou et al., 2020) or unpaired image captioning (Gu et al., 2019; Feng et al.,
2019) generalize poorly to this task since the NOC task covers a larger variety of visual concepts,
where nearly 400 object classes barely have any associated training captions.

Existing work of novel object captioning typically rely on the object detection results to either ﬁll
in the generated slotted sentences (Lu et al., 2018; Wu et al., 2018) or learn the visual vocabulary
for novel objects directly (Hu et al., 2020). However, these methods do not consider the semantics
and linguistics of the entire sentences comprehensively. Speciﬁcally, most existing works do not
exhibit the abilities in assuring the produced captions with correct and rich visual content, or with
sufﬁcient natural and ﬂuent expression. In other words, the three aspects or challenges of NOC still
need to be addressed. First of all, ﬂuency is expected in the output caption which is linguistically
natural and grammatically correct. Fidelity reﬂects the novel objects to be described, while adequacy
encourages output captions to properly capture the visual concept of the input image.

Unfortunately, the above properties cannot be easily achieved by standard image captioning mod-
els due to the absence of ground truth caption annotations. In this paper, we revisit two popular
pre-trained visual/language models of BERT (Devlin et al., 2018) and CLIP (Radford et al., 2021),

1

Under review as a conference paper at ICLR 2022

Figure 1: Overview of VLAF2 captioning model. (a) Leveraging linguistic ability from BERT for
describing novel objects. (b) Preserving adequacy and ﬁdelity of novel object captions via CLIP.

and we provide the associated connection to ﬂuency, ﬁdelity, and adequacy for NOC. As shown in
Fig. 1, we present VLAF2 for learning Visual-Linguistic Adequacy, Fidelity, and Adequacy, which
leverages the intrinsic knowledge of BERT and CLIP for utilizing linguistics observed from cap-
tions for describing visual information of images with novel objects. With BERT pre-trained to
excel at various linguistic tasks and CLIP to associate visual-language data at instance levels, we
provide insights to these models and present technical details on how such models can be utilized
for performing NOC, while the goals of adequacy, ﬁdelity and ﬂuency can be jointly achieved.

For the evaluation part of our work, we conduct extensive experiments on benchmark nocaps
datasets, conﬁrming that our model is able to produce state-of-the-art results in terms of metrics
linking to adequacy, ﬁdelity, and ﬂuency. In addition, by a variety of ablation studies, we further
provide analysis on how BERT and CLIP are utilized and thus be preferable in solving NOC tasks.

2 CONNECTING FLUENCY, FIDELITY AND ADEQUACY WITH BERT AND CLIP

We ﬁrst discuss how ﬂuency, ﬁdelity, and adequacy can be fundamentally and technically related to
the visual/language models of BERT (Devlin et al., 2018) and CLIP (Radford et al., 2021), which
will be utilized in our proposed learning framework. For caption ﬂuency, one expects that the im-
age caption output to be linguistically natural and grammatically correct. It not only requires to
capture the co-occurrence of novel-object vocabularies, the associated collocations such as verbs
or modiﬁers are expected to be properly utilized. We deﬁne the context containing novel-object
vocabularies as ˜y and the associated collocations as ˆy. Fluency can be deﬁned as how many col-
locations are observed by captioning models given a novel image context p(ˆy|˜y). This is exactly
the masked language modeling (MLM) objective adopted in BERT, without the log function explic-
itly calculated. This is the reason why we adopt BERT to learn the co-occurrence of novel-object
vocabularies and their associated collocations to improve the linguistic ﬂuency.

We now relate ﬁdelity and adequacy in image caption outputs to the model of CLIP. We start by
deﬁning objects appearing in the caption as X , and objects mentioned by the captions as Y. Relevant
objects (X , Y) are deﬁned as objects that are both included in the images and described by the
captions. Fidelity assesses whether the visual content presented in the produced caption is correct,
and can be deﬁned as the fraction of relevant objects among objects in captions p(X |Y) = p(X ,Y)
p(Y) .
Similarly, adequacy evaluates whether sufﬁcient visual details have been expressed by captions, and
we can deﬁne it as the fraction of relevant objects among objects in images p(Y|X ) = p(X ,Y)
p(X ) .
Based on the observation of Friedman (2017), these two quantities p(X |Y) and p(Y|X ) represent
the association between X and Y. While CLIP is trained to associate image and text data at the
instance level, it can be further applied for guiding image captioning models for NOC, boosting the
desirable ﬁdelity and adequacy.

3 METHOD

Before presenting the learning framework of VLAF2 for novel object captioning, we determine the
notations and settings for the sake of completeness. Given a small set of caption-labeled images
Xl, the corresponding captions Yl, as well as a large set of uncaptioned images Xu, our goal is to
generate the associated captions ˆYu for Xu using the captioning model Cθ, where θ is the parameters
of captioning model C. To achieve this, we propose a two-stage learning framework of VLAF2,
guided by two pre-trained visual-language models of BERT (Devlin et al., 2018) and CLIP (Radford

2

Captioning ModelBERTa dirt road going through a lush green hillside.a dirt road wound through a lush green hillside.a dirt road MASK through a lush green hillside.update(a)CLIPa blue shirt with a glove and a tie on ittwo glove hands and a red scarf on a blue background.Captioning Model0.550.75updatea blue shirt with a glove and a tie on ittwo glove hands and a red scarf on a blue background.(b)samplingargmaxUnder review as a conference paper at ICLR 2022

Figure 2: Learning to caption novel objects with linguistic ﬂuency. For caption-labeled image xl, we
impose the sequence-to-sequence objective Ls2s for training. For uncaptioned image xu, we exploit
u, and the reﬁned caption is denoted as ˆyb
BERT to improve the wordings of the generated caption ˆyc
u.

et al., 2021). Note that neither BERT nor CLIP have observed Xu during training. An overview of
our method is shown in Fig. 1. While our captioning model determines the labels of novel object
images using pretrained VIVO (Hu et al., 2020), our major technical contribution lies in how we
leverage visual-linguistic information from BERT to CLIP, realizing the goal of producing novel
object captions with sufﬁcient ﬂuency, ﬁdelity and adequacy.

Take Fig. 1(a) for example, when the captioning model generates a caption for a given image,
VLAF2 would regularize and substitute the verb wound for going, ensuring the corresponding lin-
guistic ﬂuency. Followed by next training stage in which we randomly samples multiple captions
from the same image with reward captions properly designed, VLAF2 would further produce cap-
tions with improved ﬁdelity and adequacy. For example, in Fig. 1(b), the caption in the lower box
accurately describes the object red scarf and thus receives a higher reward from CLIP. This encour-
ages the model to correctly describe the visual content and capture the associated visual concept of
the input image. How the pre-trained models of BERT and CLIP would guide the learning of our
VLAF2 will be detailed in the following subsections.

3.1 DESCRIBING NOVEL OBJECTS WITH LINGUISTIC FLUENCY

By observing image-caption pairs (xl, yl), the image captioning model in Fig. 1(a) would learn the
visual grounding (i.e., localization of known objects and referring their expressions) as well the
linguistics of captions. With uncaptioned images Xu containing novel objects, we adopt BERT to
reﬁne the wordings for captions containing novel objects, followed by CLIP to assess the quality of
the resulting caption outputs.

While initialized by pre-trained models of VIVO (Hu et al., 2020), our captioning model in Fig. 1(a)
would recognize images with novel objects but lack sufﬁcient ability in “describing” them in terms
of captions. To solve this problem, we impose a conventional sequence-to-sequence objective Ls2s,
which requires supervision of image-caption pairs (xl, yl). That is, we have

Ls2s = CrossEntropy(ˆyl, yl),

(1)

where ˆyl = Cθ(xl) denotes the predicted caption. As for uncaptioned images Xu, while we do
not observe ground truth caption for images containing novel objects, the collocations of the asso-
ciated novel objects would be discovered via exploiting the intrinsic knowledge of BERT. Thus, the
linguistic ﬂuency of resulting captions can be further improved.

1, wc

1, wc

2, ..., wc

2, ..., wm

T }, where wc

We now detail the learning process for the aforementioned uncaptioned image data. As illustrated in
Fig. 1, given an uncaptioned image xu, the captioning model Cθ generates a caption ˆyc
u = Cθ(xu),
ˆyc
u = {wc
i denotes the ith word and T is the caption length. The superscript
c represents it is generated by our captioning model. We then obtain a masked caption ˆym
u =
{wc
T } with m indicating the mask index, by randomly masking out the words
in the caption. We note that, we do not mask nouns in the above process, since they are viewed as
relating to objects grounded in the visual content, and language models like BERT are not designed
to handle such information. Finally, BERT takes the masked sequence ˆym
u as input and recovers
the masked word conditioned on the semantics of the entire sentence, producing the reﬁned caption
u = BERT(ˆym
ˆyb
T }.

M , ..., wc

M , ..., wc

u = {wc

2, ..., wb

u ), ˆyb

1, wc

3

Captioning Model Uncaptioned imageCaptioned imagea dirt road going through a lush green hillside.two laptops are sitting next to each other on a table next to a bunch of wiresGT Captions    =BERTa dirt road wound through a lush green hillside.maskeda dirt road wound through a lush green hillside.CLIP0.7reliable?Under review as a conference paper at ICLR 2022

Figure 3: Learning to caption novel objects with improved visual-linguistic adequacy and ﬁdelity.
For caption-labeled image xl, we perform SCST training using CIDEr as reward. The sampled cap-
tion ˆys
d will be rewarded by CLIP if it has higher cross-modal association than the greedy-decoded
baseline ˆyg
d. The superscript d indicates the source of the image. Additionally, we regularize our
model with rrep to avoid redundant and repetitive captions.

It is worth pointing out that, however, not every word substitution from BERT is semantically cor-
rect. This is the reason why we require CLIP to validate each replacement output. Speciﬁcally, if
the reﬁned caption comprises a more accurate and associated word that human generally uses to
describe the scene, then a higher CLIP score would be obtained (than that of the original caption).
Thus, we propose the objective function LBERT for learning wordings, which calculates the cross-
entropy loss for the replaced words, gated by the comparison of the CLIP scores of the two captions.
More precisely, LBERT is derived as:

LBERT = g · CrossEntropy(wc

M , wb

M ),

g =

(cid:26)0
1

if CLIP(xu, yb
if CLIP(xu, yb

u) ≤ CLIP(xu, yc
u)
u) > CLIP(xu, yc
u)

,

(2)

where CLIP(x, y) calculates the association between an image x and its caption y, indicating how
well the captions match the images. More details can be referred to Radford et al. (2021).

3.2 LEARNING NOVEL OBJECT CAPTIONS WITH FIDELITY AND ADEQUACY

Preserving the adequacy and ﬁdelity of captions for images with novel objects is another challenging
task. Recall that, as discussed in Sec. 2, ﬁdelity veriﬁes the correctness of visual content presented in
the generated caption, while adequacy assesses whether sufﬁcient visual details have been expressed
in it. Conventional sequence-to-sequence model training with cross-entropy loss might not reﬂect
the above desirable properties. This is because that, models trained with word-by-word supervision
tend to imitate the sentence patterns of the training images instead of relating caption data to the
visual content, leading to less optimal caption generation.

To tackle the above issues, we utilize the captioning evaluation metric of CIDEr as the reward in
our learning framework, since it encourages the generated caption to be consistent with that of the
human annotated ones in the word level. However, while CIDEr can be easily computed for caption-
ing labeled images Xl, it cannot be explicitly calculated for captioning uncaptioned images Xu due
to the absence of ground-truth captions. Instead, we propose to optimize for ﬁdelity and adequacy
of the generated captions via the association between images and captions calculated by CLIP, en-
couraging captions which precisely describe the objects with plentiful visual details, as discussed in
Sec. 2. However, we observe that captioning models would achieve improved association by simply
repeating the same object that occurs in the image, which undermines the linguistic ﬂuency of the
captions. For example, the image caption “a group of cans of soda and other items on a table” can
be replaced by “a pile of cans and bottles of soda on a counter with cans of cans” with a higher
CLIP score. Therefore, we additionally impose a repetition penalty to avoid such trivial solutions.
We now describe how the rewards are calculated and how do we update the captioning models.

Rewards for the generated captions. In order to realize the above properties, we design the fol-
lowing rewards to reﬂect the quality of the generated captions. For image-caption pairs (xl, yl), we
directly calculate the CIDEr reward for the predicted caption ˆyl (i.e., rCIDEr = CIDEr(ˆyl, yl)). To
encourage the generated captions for X = Xl ∪ Xu with ﬁdelity and adequacy, we exploit CLIP to
compute the association between X = Xl ∪ Xu and ˆY = ˆYl ∪ ˆYu (i.e., rCLIP = CLIP(x, ˆy)).

4

Captioning Model Uncaptioned imageCaptioned imagetwo glove hands and a red scarf on a blue background.two laptops sitting on a table with a tableGT Captions    =two laptops sitting on a table with white wires.a blue shirt with a glove and a tie on itargmaxsamplingsamplingCLIP0.750.50.60.8regularizeregularizeUnder review as a conference paper at ICLR 2022

As for repetition penalty to preserve linguistic ﬂuency of the generated captions ˆyu =
{w1, w2, ..., wT } for xu, we formulate it as a linear assignment problem, where every word is
assigned to the most similar one in the sentence except for itself. Then, we calculate the similarity
between such pairs for each sentence. Intuitively, repetitive captions would have high similarity
scores, since the repeated words will be assigned to the exact same words. We deﬁne the assignment
ˆα as the one that maximizes the average pairwise similarity of a sentence. Thus, we have:

ˆα = arg max

α

1
T

T
(cid:88)

C(wi, wα(i)),

(3)

i=1
where α(i) is the index of the word assigned to the i-th word in the caption, and C(wi, wj) is the
cosine similarity between the GloVe (Pennington et al., 2014) word representation of two words.
Since a desirable captioning model would encourage captions with low repetition (i.e., low average
pairwise similarity), the reward for repetition penalty is deﬁned as follows:

rrep = 1 −

1
T

T
(cid:88)

C(wi, w ˆα(i)),

(4)

i=1
Note that we do not calculate repetition penalty for ˆYl since they are regularized by the afore-
mentioned CIDEr rewards. With the above discussions, the total reward for caption-labeled data
would be r(ˆyl) = rCIDEr(ˆyl, yl) + rCLIP(xl, ˆyl), and the total reward for uncaptioned data would be
r(ˆyu) = rCLIP(xu, ˆyu) + rrep(ˆyu).

Back-propagation via reinforce algorithm. Unfortunately, computation of the aforementioned
rewards is non-differentiable. Thus, we adopt reinforce algorithm (Williams, 1992) to optimize the
learning of our model. As shown in Fig. 1, for an image x we randomly sample the caption ˆys
from the word distribution and use greedy decoding to obtain the baseline result ˆyg. If the sampled
captions possess higher linguistic ﬂuency or cross-modal association than the baseline caption, they
will be encouraged by positive rewards and vice versa. We follow Rennie et al. (2017); Liu et al.
(2017; 2018) and deﬁne the objective function as follows:

d) − r(ˆyg
∇θLRL(θ) ≈ −(r(ˆys
(cid:26)(rCIDEr(ˆyd, yd) + rCLIP(xd, ˆyd)

rCLIP(xd, ˆyd) + rrep(ˆyd)

d),

d))∇θ log pθ(ˆys
if xd ∈ Xl
if xd ∈ Xu

,

(5)

r(ˆyd) =

where d indicates the source of the image, θ being the parameters of captioning model, and pθ(ˆys)
represents the predicted word logits for the generated captions. With the objective functions deﬁned
in equations (1), (2), and (5), our captioning model Cθ can be trained accordingly.

4 EXPERIMENT

Datasets & Implementation Details. The training data for the nocaps benchmark comprises the
Open Images V4 (Kuznetsova et al., 2020) object detection training set (1.7M images annotated
with bounding boxes for 600 object classes), plus the image-caption pairs from the COCO Captions
2017 (Chen et al., 2015) training set (118K images containing 80 object classes). No additional
image-caption pairs are provided for training. We refer the images from the Open Images dataset to
the uncaptioned images Xu we deﬁne in Sec. 3, and the image-caption pairs from COCO Captions
are deﬁned as (Xl, Yl). We only use the Open Images datasets during VIVO pre-training but lever-
age both datasets for training as described in Sections 3.1 and 3.2. We evaluate our model on the
validation and test set of nocaps, which comprises 4500 and 10600 images from the Open Images
validation and test sets, respectively. For the architecture of our captioning model, we follow (Hu
et al., 2020; Li et al., 2020; Zhang et al., 2021) to use a BERT-base (Devlin et al., 2018) model.
As for CLIP and BERT, we directly exploit the pre-trained models released by their authors. The
architecture for BERT is BERT-Large, and the version for CLIP is ViT/B-32. Due to page limits,
hyperparameters and other training details can be found in the Appendix A.

4.1 EVALUATION METRICS

CIDEr. Similar to evaluation metrics (Papineni et al., 2002; Lin, 2004; Banerjee & Lavie, 2005)
for NLP tasks, Consensus-based Image Description Evaluation (CIDEr) calculates the similarity

5

Under review as a conference paper at ICLR 2022

Table 1: Quantitative results on nocaps. The numbers before/after slashes denote scores derived
without/with Constrained Beam Search (CBS).

Method

UpDown
OscarB
OscarL
OscarB + VIVO
VinVL
VinVL + VIVO
Human
Ours

UpDown
OscarB
OscarL
OscarB + VIVO
VinVL
VinVL + VIVO
Human
Ours

in-domain

near-domain

out-of-domain

overall

CIDEr

SPICE

- / 79.3
- / 83.4
- / 85.4
- / 92.2
97.9 / 96.8
95.8 / 94.8
84.4
102.8 / 101.4

- / 76.0
- / 81.3
- / 84.8
- / 89.0
93.0 / 93.8
94.5 / 84.4
80.6
101.7 / 90.4

- / 12.4
- / 12.0
- / 11.9
- / 12.9
13.2 / 13.5
13.3 / 13.3
14.3
14.8 / 15.1

- / 11.8
- / 11.9
- / 12.1
- / 12.9
13.3 / 13.3
13.1 / 12.8
15.0
15.0 / 14.3

CIDEr

SPICE
Validation Set

- / 73.8
- / 81.6
- / 84.0
- / 87.8
89.2 / 90.7
90.5 / 91.4
85.0
97.9 / 96.8

- / 74.2
- / 79.6
- / 82.1
- / 87.8
84.7 / 89.0
90.9 / 86.0
84.6
95.7 / 91.0

- / 11.4
- / 12.0
- / 11.7
- / 12.6
12.9 / 13.1
12.8 / 13.0
14.3
14.4 / 14.5

Test Set

- / 11.5
- / 11.9
- / 11.5
- / 12.6
12.7 / 12.7
12.9 / 12.6
14.7
14.4 / 14.0

CIDEr

SPICE

CIDEr

SPICE

- / 71.7
- / 77.6
- / 80.3
- / 87.5
68.4 / 87.4
77.1 / 88.7
95.7
86.3 / 95.4

- / 66.7
- / 73.6
- / 73.8
- / 80.1
64.0 / 66.1
73.9 / 77.9
91.6
78.9 / 82.5

- / 9.9
- / 10.6
- / 10.0
- / 11.5
10.8 / 11.6
11.1 / 11.6
14.0
12.5 / 12.9

- / 9.7
- / 10.6
- / 9.7
- / 11.1
10.9 / 10.9
11.2 / 11.3
14.2
12.1 / 12.2

- / 74.3
- / 81.1
- / 83.4
- / 88.3
86.1 / 90.9
88.6 / 91.4
87.1
96.3 / 97.2

- / 73.1
- / 78.8
- / 80.9
- / 86.6
82.0 / 85.5
88.3 / 84.4
85.3
93.5 / 89.4

- / 11.2
- / 11.7
- / 11.4
- / 12.4
12.5 / 12.8
12.5 / 12.7
14.2
14.1 / 14.2

- / 11.2
- / 11.7
- / 11.3
- / 12.4
12.4 / 12.5
12.6 / 12.4
14.6
14.1 / 13.7

Table 2: Quantitative comparisons on caption ﬂuency, ﬁdelity and adequacy. Note that BLEU@4
(B@4) and CIDEr (C) are utilize for describing ﬂuency, object precision (P) for ﬁdelity, object recall
(R) for adequacy and object F1 scores (F1) for overall cross-modal association.

Method

in-domain

near-domain

out-of-domain

B@4

C

P

R

F1

B@4

C

P

R

F1

B@4

C

P

VinVL

32.6

VinVL+VIVO 31.2

Ours

35.9

63.2

59.8

68.0

59.2

56.0

58.2

40.8

42.2

45.6

48.3

48.1

51.3

30.5

30.1

32.2

58.3

57.3

60.8

22.8

28.5

39.9

32.6

36.3

41.0

26.8

32 .0

40.4

29.6

27.3

30.1

48.8

45.6

50.2

48.4

49.0

51.3

R

25.6

27.3

30.5

F1

33.5

35.1

38.3

between the reference and generated caption by word n-gram overlap in a rule-based manner. To
capture human consensus for image captioning evaluation, it introduces the tf-idf weight to reduce
the matching weight of the n-grams that are common in all image captions.

SPICE. Semantic Propositional Image Caption Evaluation (SPICE) (Anderson et al., 2016b)
matches the semantics between sentences, such as objects, relations, and attributes of objects.
Speciﬁcally, it converts sentences into semantic scene graphs, which allows evaluation to break
grammatical constraints and focuses on propositional semantic content. It reﬂects the accuracy of
the visual content and considers less about linguistic properties.

Fluency. To quantitatively evaluate ﬂuency, we remove the effect of the visual information and focus
on the quality of linguistic properties in the conventional caption evaluation metrics. Speciﬁcally,
we remove all the objects and nouns from the captions and report BLEU@4 (Papineni et al., 2002)
and CIDEr scores calculated by the removed version of reference and candidate captions. Note that
the ﬂuency experiment is conducted on a subset of the nocaps validation set, which contains 1000
images whose caption annotations are available on the ofﬁcial website of the nocaps dataset.

Fidelity & Adequacy. Fidelity and adequacy evaluate how well the captions are associated with
images. As deﬁned in Sec. 2, ﬁdelity stands for the fraction of relevant objects described in cap-
tions among all the objects in captions, and adequacy is the fraction of relevant instances that were
retrieved. These two properties are analogous to the deﬁnition of precision and recall, respectively.
Therefore, we extract the objects mentioned in the captions and the ground-truth objects in the im-
ages and calculate the precision (for ﬁdelity), recall (for adequacy), and F1 (for overall association)
scores. The experiment is performed on the validation set of nocaps.

Following Agrawal et al. (2019), we further split the dataset into three subsets for evaluation: in-
domain images only contain the seen objects that have been described in the training captions, out-
of-domain images only with unseen (i.e., novel) objects presented, and near-domain ones containing
both seen and unseen objects.

6

Under review as a conference paper at ICLR 2022

Table 3: Analyses on BERT, CLIP, and repetition penalty for NOC using nocaps validation set. Note
that BERT mainly beneﬁts the linguistic ﬂuency with improved CIDEr, and CLIP is desirable for
preserving visual semantics with increased SPICE.

Method

Ours
Ours w/o LBERT
Ours w/o rCLIP
Ours w/o rrep

in-domain

near-domain

out-of-domain

overall

CIDEr
102.77
99.13
101.12
96.73

SPICE CIDEr
97.90
14.83
94.72
14.41
94.07
13.80
14.83
89.64

SPICE CIDEr
86.33
14.40
84.47
14.11
80.54
13.35
81.87
14.12

SPICE CIDEr
96.25
12.54
93.27
12.38
92.33
11.94
89.08
12.38

SPICE
14.10
13.81
13.14
13.88

Table 4: Analyses on BERT and CLIP for improving caption ﬂuency, ﬁdelity and adequacy. Note
that BERT beneﬁts ﬂuency metrics of BLEU@4 (B@4) and CIDEr (C), while CLIP focusing on
cross-modal association boosts metrics of object precision (P), recall (R), and F1 scores (F1).

Method

Ours

Ours w/o LBERT

Ours w/o rCLIP

in-domain

near-domain

out-of-domain

B@4

C

P

R

F1

B@4

C

P

R

F1

B@4

C

P

R

F1

35.9

32.8

33.2

68.0

64.8

65.9

58.2

58.1

58.8

45.6

51.3

40.9

42.1

48

49.1

32.2

30.4

31.9

60.8

60.6

60.7

39.9

41.0

40.4

41

35.8

39.0

37.7

40.0

36.8

30.1

27.9

28.7

50.2

49.1

49.8

51.3

51.6

51.6

30.5

38.3

27.9

27.4

36.2

35.8

4.2 QUANTITATIVE ANALYSIS

For performance comparisons, we choose UpDown (Agrawal et al., 2019) without SCST optimiza-
tion (Rennie et al., 2017) and Oscar (Li et al., 2020) as baselines, as well as VinVL (Zhang et al.,
2021) that achieves SOTA results on the benchmark of nocaps. In addition, VIVO (Hu et al., 2020) is
a pre-training technique for captioning models, allowing them to recognize the novel objects. Since
VinVL did not report the numbers with Constrained Beam Search (CBS) exploited during infer-
ence (CBS is known to improve model performance on out-of-domain data), we reproduce VinVL
following details stated in the original paper. For more details please refer to Appendix A. For a
comprehensive comparison, we conduct experiments on the validation and test set of nocaps. In
addition, we compare our method with VinVL in ﬂuency, ﬁdelity, and adequacy to demonstrate the
improvement in terms of these properties. We also evaluate on the COCO Caption dataset and report
in Appendix B.

The nocaps datasets. The results on nocaps are shown in Table 1. From this table, we see that
our model performed favorably against baselines and SOTAs across different metrics. We observe
that CBS slightly decreased model performance on seen object captioning, since it forces captioning
models to describe the detected objects without considering detection correctness. Nevertheless, for
near or out-of-domain images, CBS still beneﬁts the captioning performances. It is worth noting that
our model largely increased the performance in SPICE score for every data domain, which veriﬁes
that our method is able to generate captions with the improved image-language association.

Fluency, ﬁdelity, and adequacy. As described in Sec. 4.1, we design additional experiments for
evaluating ﬂuency, ﬁdelity, and adequacy and report the numbers in Table 2. For ﬂuency, we remove
all the objects and nouns in the captions since they relate less to the linguistics of the captions. We
then calculate the BLEU@4 (B@4) and CIDEr (C) scores for the captions after removal. For ﬁdelity
and adequacy, they indicate that captions should accurately (high precision) describe sufﬁcient (high
recall) visual details. Therefore, we report the object precision and recall in this table, and object F1
scores represent the overall association between captions and images. One can see that our method
surpass previous methods by a visible margin on all tasks except for in-domain object precision.,
which further veriﬁes our model improves novel object captioning on ﬂuency, ﬁdelity, and adequacy.

4.3 REMARKS ON BERT AND CLIP FOR CAPTION FIDELITY, ADEQUACY AND FLUENCY

We conduct ablation studies on the nocaps validation set, with the aim to verify the necessity of
integrating BERT and CLIP for learning NOC models. Following the same evaluation procedures
described in Sec. 4.1, we discuss the contributions of these models in terms linguistic and semantic
level metrics in Tables 3 and 4. Detailed ablation analysis of every objective can be further found in
Appendix B.2.

7

Under review as a conference paper at ICLR 2022

Figure 4: Example results and comparisons for image captions produced by VinVL and ours in
terms of ﬂuency, ﬁdelity and adequacy. Note that both utilize VIVO for novel object detection.

BERT. As shown in Table 3, the captioning model without BERT would observe a performance drop
in CIDEr for linguistic ﬂuency, but such drops for the metric of SPICE (related to visual content)
would be less signiﬁcant. Similarly, as observed in Table 4, removing BERT would result in the
lowest BLEU and CIDEr scores. These results conﬁrm our motivation and model design discussed
in Sec. 2, since BERT is utilized to improve caption quality at the linguistics level.

CLIP. As shown in Table 3, the model without CLIP showed signiﬁcant drops in captioning metrics
of CIDEr and SPICE. However, the performance decrease in SPICE is expected, since CLIP is
particularly deployed in our framework for preserving visual content in captions. As for CIDEr,
its decrease is mainly due to the deterioration of missing visual content in captions. This is also
conﬁrmed by Table 4, in which mainly the metrics reﬂecting ﬁdelity and adequacy (i.e., at the visual
semantics level) would observe signiﬁcant drops for model trained without CLIP rewards.

Repetition penalty. Recall that, in Sec. 3.2, this penalty is to alleviate association between images
and captions with redundant visual information. As seen in Table 3, the model without this penalty
observed a signiﬁcant performance drop in CIDEr scores. We did not see such trends for SPICE.
This is because that, repetitive words in captions mainly violate linguistic structures rather then
semantic accuracy, and thus the performance related to linguistic ﬂuency would be more sensitive to
the deployment of this penalty.

4.4 QUALITATIVE ANALYSIS

We now empirically show captions in Fig. 4, which are generated by our model and VinVL, with
both pretrained from VIVO for novel object detection. In this ﬁgure, wordings that are less accurate
or incorrectly describe the associated visual content are marked in bold. And, our wording improve-
ments are highlighted in red. From this ﬁgure, one can see that for ﬂuency, our model generated

8

a man in a suit and tie standing in a room.FluencyFidelityAdequacyVinVLOursVinVLOursFluencyFidelityAdequacya ceiling fan in a room with a windowa blue blanket and a glove on a blue shirt.a ceiling fan hanging from the ceiling in a room.two glove hands and a red scarf on a blue background.a man in a suit and tie standing at a podium in a room.a group of women standing on a field with a football.a woman sitting at a tennis table with a racket.a man and a woman playing an accordion.a group of girls in football uniforms posing for a picture.a woman in a wheelchair playing tennis at a table.a couple of people sitting in a church playing an accordion.Under review as a conference paper at ICLR 2022

vivid captions with more proper wordings. Take the upper-left image for example, our model partic-
ularly described “fan hanging from the ceiling in a room” instead of “fan in a room”. As for ﬁdelity,
our model is designed to capture the visual content in an image. Speciﬁcally, take the second im-
age in the ﬁrst row for example, we correctly described the number of gloves and the novel object
red scarf, while VinVL failed to do so. As for adequacy, take the bottom-right image for example,
our model was able to recover visual details in the image (i.e., “people playing the accordion” and
“sitting in a church”). For more qualitative examples, please refer to Appendix B.5.

5 RELATED WORK

Image captioning. Recent progress of image captioning focuses on different model architectures
and learning methods. Gao et al. (2019); Huang et al. (2019); Wang et al. (2019); Guo et al. (2020);
Pan et al. (2020); Cornia et al. (2020) design different attention mechanisms for image caption-
ing. Rennie et al. (2017); Li et al. (2019); Yang et al. (2020) adopt reinforcement learning to improve
the performance. On the other hand, some researchers consider more challenging settings, such as
partially supervised (Liu et al., 2018; Kim et al., 2019) or unpaired image captioning (Gu et al.,
2019; Feng et al., 2019). However, these methods are restricted to the assumption that the unpaired
images and captions share the same set of object class, and the number of object class is limited as
well, which make them inapplicable to our task.

Novel object captioning. Previously, novel object captioning approaches (Anderson et al., 2016a;
2018; Hendricks et al., 2016) were only tested on a restrictive dataset with only eight novel object
classes held out from the COCO dataset. Their extensions to large-scale image data with various
novel objects are not sufﬁciently studied. Recent studies mainly rely on object detection results to
improve the performance on novel object captioning. Lu et al. (2018); Wu et al. (2018) generate
slotted caption templates, which are later ﬁlled in with visual concepts identiﬁed by object detec-
tors. Yao et al. (2017) exploits a copying mechanism to assemble words corresponding to object
detector predictions to generate captions. Similarly, Constrained Beam Search (CBS) (Anderson
et al., 2016a) is an architecture-agnostic decoding algorithm that can be exploited during inference
to enforce the inclusion of novel object classes in the captions. Instead of explicitly using detection
results, Hu et al. (2020) learns the relationship between image and text by aligning object detec-
tion tags with their corresponding image region features. Recently, Wang et al. (2021a) indicate
that a desirable caption should comprise properties of ﬂuency, ﬁdelity, and adequacy. Nevertheless,
most existing NOC approaches are not designed to handle language expression and cross-modal
association with the above properties preserved.

Vision and Language Pre-training (VLP). Existing VLP works can be classiﬁed into two streams.
One is based on the dual-encoder architecture, such as CLIP (Radford et al., 2021) and ALIGN (Jia
et al., 2021), utilizing features encoded in each modality followed by contrastive learning (Oord
et al., 2018) for alignment purposes. On the other hand, models like Zhou et al. (2020); Li et al.
(2020); Zhang et al. (2021); Yang et al. (2021) exploit multiple cross-attention layers to learn the
relationship between images and text, and show impressive performances on image-text matching
tasks. Due to the efﬁciency of the former models in handling data across modalities, we adopt CLIP
for cross-modal association in this paper.

6 CONCLUSION

A visual-linguistic learning framework with improve d adequacy, ﬁdelity, and ﬂuency (VLAF2) is
presented for novel object captioning. We fundamentally quantify the above properties and relate
them to the models of BERT and CLIP. We propose objectives and rewards reﬂecting the desir-
able linguistic ﬂuency and visual semantics for NOC. Guided by BERT, our model learns to reﬁne
wordings of novel objects. Via reinforce algorithms, we have CLIP-based rewards assess the cor-
rectness of visual content described in the generated caption. Empirically, we showed that our model
achieved SOTA results on the nocaps benchmark. We further provided analyses on both BERT and
CLIP, verifying the necessity of their integration for learning novel object captions. A future di-
rection of this work would be utilizing both captioned and uncaptioned images with self-supervised
learning strategies for training NOC models.

9

Under review as a conference paper at ICLR 2022

ETHICS STATEMENT

We acknowledge that all authors of this work have read and commit to adhering to the ICLR code
of ethics. Our work has no ethical concern.

REPRODUCIBILITY STATEMENT

For reproducing our results in the experiment section, codes and training/testing scripts are provided
in the supplementary materials.

REFERENCES

Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra,
Devi Parikh, Stefan Lee, and Peter Anderson. nocaps: novel object captioning at scale.
In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8948–8957,
2019.

Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Guided open vocabulary

image captioning with constrained beam search. arXiv preprint arXiv:1612.00576, 2016a.

Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propo-
In European conference on computer vision, pp. 382–398.

sitional image caption evaluation.
Springer, 2016b.

Peter Anderson, Stephen Gould, and Mark Johnson. Partially-supervised image captioning. arXiv

preprint arXiv:1806.06004, 2018.

Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved
correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic
evaluation measures for machine translation and/or summarization, pp. 65–72, 2005.

Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and
C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv
preprint arXiv:1504.00325, 2015.

Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, and Rita Cucchiara. Meshed-memory trans-
former for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 10578–10587, 2020.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Yang Feng, Lin Ma, Wei Liu, and Jiebo Luo. Unsupervised image captioning. In Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4125–4134, 2019.

Jerome H Friedman. The elements of statistical learning: Data mining, inference, and prediction.

springer open, 2017.

Lianli Gao, Kaixuan Fan, Jingkuan Song, Xianglong Liu, Xing Xu, and Heng Tao Shen. Deliberate
attention networks for image captioning. In Proceedings of the AAAI Conference on Artiﬁcial
Intelligence, volume 33, pp. 8320–8327, 2019.

Jiuxiang Gu, Shaﬁq Joty, Jianfei Cai, Handong Zhao, Xu Yang, and Gang Wang. Unpaired image
captioning via scene graph alignments. In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pp. 10323–10332, 2019.

Longteng Guo, Jing Liu, Xinxin Zhu, Peng Yao, Shichen Lu, and Hanqing Lu. Normalized and
geometry-aware self-attention network for image captioning. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 10327–10336, 2020.

10

Under review as a conference paper at ICLR 2022

Lisa Anne Hendricks, Subhashini Venugopalan, Marcus Rohrbach, Raymond Mooney, Kate Saenko,
and Trevor Darrell. Deep compositional captioning: Describing novel object categories without
In Proceedings of the IEEE conference on computer vision and pattern
paired training data.
recognition, pp. 1–10, 2016.

Xiaowei Hu, Xi Yin, Kevin Lin, Lijuan Wang, Lei Zhang, Jianfeng Gao, and Zicheng Liu. Vivo:
Surpassing human performance in novel object captioning with visual vocabulary pre-training.
arXiv e-prints, pp. arXiv–2009, 2020.

Lun Huang, Wenmin Wang, Jie Chen, and Xiao-Yong Wei. Attention on attention for image cap-
In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp.

tioning.
4634–4643, 2019.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan
Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning
with noisy text supervision. arXiv preprint arXiv:2102.05918, 2021.

Dong-Jin Kim, Jinsoo Choi, Tae-Hyun Oh, and In So Kweon. Image captioning with very scarce su-
pervised data: Adversarial semi-supervised learning approach. arXiv preprint arXiv:1909.02201,
2019.

Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab
Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4.
International Journal of Computer Vision, 128(7):1956–1981, 2020.

Nannan Li, Zhenzhong Chen, and Shan Liu. Meta learning for image captioning. In Proceedings of

the AAAI Conference on Artiﬁcial Intelligence, volume 33, pp. 8626–8633, 2019.

Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong
Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language
tasks. In European Conference on Computer Vision, pp. 121–137. Springer, 2020.

Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization

branches out, pp. 74–81, 2004.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
In European

Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context.
conference on computer vision, pp. 740–755. Springer, 2014.

Siqi Liu, Zhenhai Zhu, Ning Ye, Sergio Guadarrama, and Kevin Murphy.

Improved image cap-
In Proceedings of the IEEE international

tioning via policy gradient optimization of spider.
conference on computer vision, pp. 873–881, 2017.

Xihui Liu, Hongsheng Li, Jing Shao, Dapeng Chen, and Xiaogang Wang. Show, tell and dis-
criminate: Image captioning by self-retrieval with partially labeled data. In Proceedings of the
European Conference on Computer Vision (ECCV), pp. 338–354, 2018.

Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Neural baby talk. In Proceedings of the

IEEE conference on computer vision and pattern recognition, pp. 7219–7228, 2018.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predic-

tive coding. arXiv preprint arXiv:1807.03748, 2018.

Yingwei Pan, Ting Yao, Yehao Li, and Tao Mei. X-linear attention networks for image captioning.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
10971–10980, 2020.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic
evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association
for Computational Linguistics, pp. 311–318, 2002.

Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word
representation. In Proceedings of the 2014 conference on empirical methods in natural language
processing (EMNLP), pp. 1532–1543, 2014.

11

Under review as a conference paper at ICLR 2022

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.

Steven J Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. Self-critical
In Proceedings of the IEEE conference on computer

sequence training for image captioning.
vision and pattern recognition, pp. 7008–7024, 2017.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned,
hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.
2556–2565, 2018.

Guanglu Song, Yu Liu, and Xiaogang Wang. Revisiting the sibling head in object detector. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11563–
11572, 2020.

Sijin Wang, Ziwei Yao, Ruiping Wang, Zhongqin Wu, and Xilin Chen. Faier: Fidelity and adequacy
In Proceedings of the IEEE/CVF Conference on Computer

ensured image caption evaluation.
Vision and Pattern Recognition, pp. 14050–14059, 2021a.

Weixuan Wang, Zhihong Chen, and Haifeng Hu. Hierarchical attention network for image caption-
ing. In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, volume 33, pp. 8957–8964,
2019.

Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Sim-
ple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904,
2021b.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement

learning. Machine learning, 8(3):229–256, 1992.

Yu Wu, Linchao Zhu, Lu Jiang, and Yi Yang. Decoupled novel object captioner. In Proceedings of

the 26th ACM international conference on Multimedia, pp. 1029–1037, 2018.

Xuewen Yang, Heming Zhang, Di Jin, Yingru Liu, Chi-Hao Wu, Jianchao Tan, Dongliang Xie,
Jue Wang, and Xin Wang. Fashion captioning: Towards generating accurate descriptions with
semantic rewards. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK,
August 23–28, 2020, Proceedings, Part XIII 16, pp. 1–17. Springer, 2020.

Zhengyuan Yang, Yijuan Lu, Jianfeng Wang, Xi Yin, Dinei Florencio, Lijuan Wang, Cha Zhang, Lei
Zhang, and Jiebo Luo. Tap: Text-aware pre-training for text-vqa and text-caption. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8751–8761, 2021.

Ting Yao, Yingwei Pan, Yehao Li, and Tao Mei. Incorporating copying mechanism in image cap-
tioning for learning novel objects. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pp. 6580–6588, 2017.

Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual
denotations: New similarity metrics for semantic inference over event descriptions. Transactions
of the Association for Computational Linguistics, 2:67–78, 2014.

Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and
Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5579–5588, 2021.

Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason Corso, and Jianfeng Gao. Uniﬁed
vision-language pre-training for image captioning and vqa. In Proceedings of the AAAI Confer-
ence on Artiﬁcial Intelligence, volume 34, pp. 13041–13049, 2020.

12

Under review as a conference paper at ICLR 2022

Algorithm 1: Learning to caption novel objects with linguistic ﬂuency
Input: Captioning model Cθ(·), Pre-trained BERT, and Pre-trained CLIP.
Data: Captioned image xl , the corresponding GT caption yl , uncaptioned image xu , and lr ηit.
Output: Trained Captioning model Cθ(·).

u by randomly masking words in the sentence in ˆyc

u (except for nouns);

4

1 Initialize Cθ(·);
2 for it from 1 to num iters do
ˆyl ← Cθ(xl ), ˆyc
3
Produce ˆym
u ← BERT(ˆym
ˆyb
u )
Ls2s ← CrossEntropy(ˆyl , yl )
if CLIP(xu , ˆyb

u ← Cθ(xu );

5

6

7

LBERT ← 0

else

u ) ≤ CLIP(xu , ˆyc

u ) then

LBERT ← CrossEntropy(ˆyc

u , ˆyb
u )

end
L ← Ls2s + LBERT
Update parameters: θ ← Adam(θ, ηit, ∇θL)

8

9

10

11

12

13
14 end

Algorithm 2: Learning novel object captions with ﬁdelity and adequacy
Input: Captioning model Cθ(·) and Pre-trained CLIP.
Data: Captioned image xl , the corresponding GT caption yl , uncaptioned image xu , and lr ηit.
Output: Trained Captioning model Cθ(·).

4

1 Initialize Cθ(·);
2 for it from 1 to num iters do
ˆys
l ← Cθ(xl ), ˆys
u ← Cθ(xu ) (by sampling);
3
ˆyg
l ← Cθ(xl ), ˆyg
u ← Cθ(xu ) (by greedy decoding);
Calculate rrep(ˆys
u) and rrep(ˆyg
u) by (4)
r (ˆyl) ← rCIDEr(ˆyl, yl) + rCLIP(xl, ˆyl)
r (ˆyu) ← rCLIP(xu, ˆyu) + rrep(ˆyu)
Calculate the gradient ∇θLRL(θ) ← −(r (ˆys
Update parameters: θ ← Adam(θ, ηit, ∇θLRL)

8

6

7

5

9
10 end

d) − r (ˆyg

d))∇θ log pθ(ˆys

d), d ∈ {l, u}

A IMPLEMENTATION DETAILS

Following Hu et al. (2020); Li et al. (2020); Zhang et al. (2021), we consider a BERT-base (Devlin
et al., 2018) architecture for our captioning model. Given an image, the captioning model jointly
takes the image region features and the predicted detection tags to generate the associated caption.
We use the same region features as VinVL (Zhang et al., 2021), which are released on their project
page. Since the object detection model Omni-detection used in previous works (Hu et al., 2020;
Zhang et al., 2021) is not available, we replace it with a publicly available model of TSD (Song
et al., 2020) to generate the object detection tag.

Reproducing our method. We perform VIVO (Hu et al., 2020) pre-training for 100 epochs with
a batch size of 1024 and a learning rate of 5 × 10−5, which are exactly the same as the parameters
stated in the VIVO paper. After that, we propose to train our model following the training process
described in Algorithm 1 to learn to caption novel objects with linguistic ﬂuency. We train our model
for 20 epochs with an effective batch size of 512 (256 caption-labeled images and 256 uncaptioned
images) and a learning rate of 1.5 × 10−5. Then, to learn novel object captions with ﬁdelity and
adequacy, we train our model as decsribed in Algorithm 2. Speciﬁcally, we train our model for 4
epochs with an effective batch size of 128 (64 caption-labeled images and 64 uncaptioned images)
and a learning rate of 2.5 × 10−6. We use 8 V100 GPUs to perform the above training algorithms.
Codes can be found in the supplementary materials.

13

Under review as a conference paper at ICLR 2022

Table 5: Image captioning evaluation results on COCO “Karpathy” test split.

VinVL
VinVL+VIVO
Ours

BLEU@4 METEOR ROUGE-L CIDEr
134.6
134.5
137.3

39.8
39.7
40.0

59.6
59.6
60.2

29.9
29.9
30.4

SPICE
23.9
23.8
24.5

Table 6: Ablation studies on nocaps validation set.

Method

Baseline (Only w/ Ls2s)
+ LBERT
+ rCIDEr
+ rCLIP
+ rrep (Ours)

in-domain

near-domain

out-of-domain

overall

CIDEr
89.07
92.46
101.19
96.73
102.77

SPICE CIDEr
83.29
13.29
85.79
13.40
95.38
13.84
14.83
89.64
97.90
14.83

SPICE CIDEr
68.77
12.61
73.21
12.92
83.24
13.44
81.87
14.12
86.33
14.40

SPICE CIDEr
81.17
10.59
84.20
11.40
93.75
12.06
89.08
12.38
96.25
12.54

SPICE
12.32
12.69
13.23
13.88
14.10

Reproducing baseline methods. For VinVL (Zhang et al., 2021), we leverage the released model on
their project page and directly inference on the nocaps dataset. However, for VinVL+VIVO (Zhang
et al., 2021), since the pre-trained model is not publicily available, we reproduce this method using
the image region features and object detection tags generated by models mentioned in the beginning
of this section to train this model. Speciﬁcally, the model is trained for 160K iterations (about 100
epochs) with a batch size of 1024 and a learning rate of 5 × 10−5, and ﬁne-tuned for 30 epochs with
a batch size of 256 and a learning rate of 5 × 10−5 using the cross-entropy loss. Last, we perform
the SCST optimization (Rennie et al., 2017) with a learning rate of 2 × 10−6 for 5 epochs to obtain
the ﬁnal model. The numbers reported in Table 1 are derived using this version of model.

B ADDITIONAL EXPERIMENTS

B.1 EXPERIMENTS ON THE COCO CAPTION DATASET

To validate that our method generalize well on the task of describing the seen objects, we conduct
experiments on the COCO Caption test set and report the numbers in Table 5. The training data for
VinVL (Zhang et al., 2021) is image caption pairs from the COCO (Lin et al., 2014) dataset. While
for VinVL + VIVO and our method, we additionally leverage the uncaptioned image from the Open
Images (Kuznetsova et al., 2020) dataset as extra data. One can see that our method outperforms the
other competitive approaches on different metrics which veriﬁes the effectiveness of our approach.

B.2 DETAILED ABLATION ANALYSIS

Table 6 lists the performances and compares contributions of the imposed objectives in our VLAF2.
The baseline model in Table 6 is only trained on the COCO Caption dataset using the sequence-to-
sequence objective. To conﬁrm our introduction of exploiting BERT to learn the associated wordings
of novel images, we apply LBERT to the baseline model, and report the results in the second row
of Table 6. The CIDEr scores improve signiﬁcantly after adopting reinforce algorithm (Williams,
1992) and using CIDEr scores of the generated captions as reward, and the results are in the third
row. One can see that the SPICE scores largely increase but the CIDEr scores slightly decrease after
the deployment of CLIP. We hypothesize that the captioning model properly captures the visual
content in images, but it describes the scene with poor linguistic ﬂuency. As the discussion in
Sec. 3.2, we attribute the performance drop to the degenerate solution of increasing the association
between the captions and the corresponding images. Note that we further consider the repetition
penalty to regularize the captioning model. The results are shown in the last row of Table 6. One can
see that this regularization slightly improves the SPICE scores but signiﬁcantly increase the CIDEr
scores. By comparing the performances listed in Table 6, we see that the full version of our VLAF2
achieved the best performance in terms of CIDEr and SPICE. Thus, the design of our VLAF2 can
be successfully veriﬁed.

14

Under review as a conference paper at ICLR 2022

Table 7: Quantitative results on the nocaps (XD) test set.

Method

UpDown (Agrawal et al., 2019)
SimVLMbase (Wang et al., 2021b)
VIVO (Hu et al., 2020)
Ours
Ours (+CC)

overall

CIDEr
73.09
94.80
100.12
96.25
102.39

SPICE
11.20
13.10
14.04
14.10
14.71

Table 8: Ablation studies of the joint-training model on nocaps validation set.

in-domain

near-domain

Method

Baseline (Only w/ Ls2s)
+ LBERT
+ rCIDEr
+ rCLIP
+ rrep (Ours)

CIDEr
96.1
99.44
109.14
103.81
110.56

SPICE
13.71
13.91
14.52
15.99
15.23

CIDEr
90.35
91.13
100.66
98.91
105.16

SPICE CIDEr
79.96
13.41
81.11
13.53
88.61
14.08
15.32
93.17
96.22
14.81

out-of-domain
SPICE
11.77
11.82
12.69
13.67
13.19

overall

CIDEr
89.07
90.29
99.43
98.45
104.12

SPICE
13.13
13.25
13.87
15.09
14.55

B.3 EXPERIMENTS ON THE NOCAPS (XD) BENCHMARK

To investigate the limits of performance on nocaps without any restraints on the training datasets,
we conduct experiments on the nocaps (XD) benchmark to verify the effectiveness of our method
when more image-caption pairs are considered. Speciﬁcally, we additionally consider Conceptual
Captions (CC) (Sharma et al., 2018) as labeled training samples Xl in our learning framework
and perform joint-training to see if extra image-caption pairs beneﬁt the model on novel object
captioning. The results are shown in Table 7. Note that both VIVO, SimVLM and our method adopt
BERT-based architecture as the backbones for captioning. As for the training set, since VIVO (Hu
et al., 2020) did not specify the dataset details for their evaluation for the nocaps (XD), we compared
our method to SimVLM, which applied a much larger web-scale dataset (1.8B image-text pairs) than
the CC dataset (3.1M pairs). Yet, our method still performs favorably against SOTAs on the nocaps
(XD) protocol and benchmark, verifying the effectiveness of our method even if more image-caption
pairs are considered.

In addition, to quantitatively show that the performance gain in Table 7 is not simply contributed by
the additional data we considered, we ablate our model on the nocaps validation set and show the
results in Table 8. We observed a similar performance trend as we reported in Table 6, where LBERT
slightly improves the CIDEr scores, and rCLIP siginiﬁcantly boost SPICE but slightly deteriorates
the CIDEr scores. One can see that the regularization rrep slightly improves the SPICE scores but
signiﬁcantly increase the CIDEr. By comparing the performances listed in Table 6 and Table 8,
we see that our design of distilling BERT to enhance ﬂuency (in terms of CIDEr) and the uses of
CLIP to encourage captions with sufﬁcient ﬁdelity and adequacy (in terms of SPICE) still function
properly when more diverse image-caption pairs are considered, verifying the design of our VLAF2.

B.4 HUMAN STUDY

To conduct human study, we randomly picked 60 images from the nocaps validation set,
and compared the captions generated by our method to those generated by the SOTA of
VinVL+VIVO (Zhang et al., 2021), and the human-annotated captions provided by the nocaps
dataset. Following the evaluation protocols used in the COCO Captioning Challenge 2015 (Lin
et al., 2014), we designed 4 different metrics and asked individuals to evaluate captions from these
aspects. The following are the four metrics we used in the experiment: M1: Is the caption generated
by human (0: machine, 1: human)? (Percentage of captions that pass the Turing Test.) M2: Rate the
correctness of the captions on a scale 1-5 (incorrect-correct): Whether the described objects or activ-
ities are correct. M3: Rate the amount of detail of the captions on a scale 1-5 (lack of details - very
detailed): Whether the caption has detailed all the objects and their attributes. M4: Rate the ﬂuency

15

Under review as a conference paper at ICLR 2022

Method
VinVL+VIVO
Ours
Human

Table 9: Human study on the nocaps validation set.
M1 (Turing Test) M2 (Fluency) M3 (Fidelity) M4 (Adequacy)

0.25
0.43
0.53

3.99
4.06
4.09

3.70
4.33
4.44

3.46
4.24
4.18

of the captions on a scale 1-5 (lack of ﬂuency-very ﬂuent): Whether the caption use phrases/words
that human generally would use to describe the scene, i.e., the caption is linguistically natural and
ﬂuent.

Speciﬁcally, M2, M3, M4 correspond to the ﬁdelity, adequacy, and ﬂuency, respectively, which
are the particular objectives desired to be achieved. We asked 24 people two answer 6 different
questionnaires, and each questionnaire contains 10 captions from each method (i.e., ours, sota, and
human caption presented in a random order). We report the results in Table 9. We see that our
method surpassed the SOTA by clear margins, while our performances were comparable to those
the human ones across different metrics. This further supports the design of our model for NOC
with sufﬁcient ﬂuency, ﬁdelity, and adequacy.

B.5 MORE QUALITATIVE RESULTS

Qualitative comparison on ﬂuency, ﬁdelity and adequacy. In this part, we provide more quali-
tative results on the nocaps validation/test set, and the results are shown in Fig. 5 and 6. Note that
wordings that are less accurate or incorrectly describe the associated visual content are marked in
bold. And, our wording improvements are highlighted in red. Take results in the bottom row of Fig.
5 for example. For the column of ﬂuency, our model particularly described the turtle being “crawl-
ing on some rocks” instead of “sitting on the top of a beach”. For ﬁdelity, our model predicted
the background preferably as “race track” instead of “street” from the prediction of VinVL model.
As for the column of adequacy, though both captions described a young men running, our model
successfully captures more details in the image (i.e., “there are number on their shirts”). For more
qualitative results, please refer to Fig. 6.

Qualitative results of some failure cases. In this part, we demonstrate some failure cases of our
VLAF2 model. We empirically observe that the failure cases mainly come from the wrong/missing
detection tags predicted by the pre-trained object detectors. To be more speciﬁc, the captioning
model largely relies on the detection tags as clues to correctly describe novel objects. Take the
result in the left-side of Fig. 7 for example, the detection model falsely recognizes the raccoon as a
squirrel, and this detection result consequently damages the caption prediction. Therefore, how to
jointly improve the detection model and captioning model is still a open question, and we leave this
problem for future research. For more failure cases, please refer to Fig. 7.

16

Under review as a conference paper at ICLR 2022

Figure 5: Example results and comparisons for image captions produced by VinVL and ours in
terms of ﬂuency, ﬁdelity and adequacy. Note that both utilize VIVO for novel object detection.

17

a woman sitting in front of a piano playing a keyboard.FluencyFidelityAdequacyVinVLOursVinVLOursFluencyFidelityAdequacya group of men cutting up watermelon in a field.a woman standing in front of a projector screen with a presentation.a group of men standing in a field with watermelon.a woman standing in front of a computer screen.a woman sitting on a piano.a large turtle crawling on some rocks in the dirt.a group of men running down a race track.a tortoiset sitting on top of a beach.a group of men running down a street.a group of young men running in a field.a group of young men running in the grass with numbers on their shirts.Under review as a conference paper at ICLR 2022

Figure 6: Example results and comparisons for image captions produced by VinVL and ours in
terms of ﬂuency, ﬁdelity and adequacy. Note that both utilize VIVO for novel object detection.

Figure 7: False captions misled by the wrong object detection tags.

18

a man riding a bike on a street with a helmet.FluencyFidelityAdequacyVinVLOursVinVLOursFluencyFidelityAdequacya yellow lamp with a light bulb on a black background.a black vase sitting on top of a table.a man riding a pink bike in the street.a woman wearing a brown hat and smiling.a woman wearing a brown jacket and boots holding a cell phone.a woman wearing a hat and a table.a woman holding a cell phone in her hand.a couple of people sitting on a red bike.a couple of people riding on a red bike.a yellow bee flying next to a bunch of purple flowers.a yellow bee sitting on top of blue flowers.a brown squirrel sitting on a tree branch.a man holding a tennis racket on a fieldMan, Football helmet, Sports uniform, Baseball glove, Baseball bata collage of pictures of dogs and a lion.Detection tagsOursTree, Monkey, squirrelDog, Carnivore, Lion, Brown bearGT tagsRaccoonlacrosse stickJaguar