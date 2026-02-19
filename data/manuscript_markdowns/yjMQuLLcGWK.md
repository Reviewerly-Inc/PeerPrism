Published as a conference paper at ICLR 2022

FP-DETR: DETECTION TRANSFORMER ADVANCED
BY FULLY PRE-TRAINING

Wen Wang1∗, Yang Cao1,2†, Jing Zhang3, Dacheng Tao4,3
1University of Science and Technology of China
2Institute of Artiﬁcial Intelligence, Hefei Comprehensive National Science Center
3The University of Sydney, 4JD Explore Academy, China
wangen@mail.ustc.edu.cn, forrest@ustc.edu.cn,
jing.zhang1@sydney.edu.au, dacheng.tao@gmail.com

ABSTRACT

Large-scale pre-training has proven to be effective for visual representation learn-
ing on downstream tasks, especially for improving robustness and generalization.
However, the recently developed detection transformers only employ pre-training
on its backbone while leaving the key component, i.e., a 12-layer transformer,
being trained from scratch, which prevents the model from above beneﬁts. This
separated training paradigm is mainly caused by the discrepancy between the up-
stream and downstream tasks. To mitigate the issue, we propose FP-DETR, a
new method that Fully Pre-Trains an encoder-only transformer and smoothly ﬁne-
tunes it for object detection via a task adapter. Inspired by the success of textual
prompts in NLP, we treat query positional embeddings as visual prompts to help
the model attend to the target area (prompting) and recognize the object. To this
end, we propose the task adapter which leverages self-attention to model the con-
textual relation between object query embedding. Experiments on the challenging
COCO dataset demonstrate that our FP-DETR achieves competitive performance.
Moreover, it enjoys better robustness to common corruptions and generalization to
small-size datasets than state-of-the-art detection transformers. Code will be made
publicly available at https://github.com/encounter1997/FP-DETR.

1

INTRODUCTION

Since the surge of deep learning-based object detector (Girshick et al., 2014), pre-training the back-
bone on large-scale datasets like ImageNet (Deng et al., 2009) and ﬁne-tuning the model on down-
stream tasks has become a well-established paradigm. Pre-training signiﬁcantly improves robust-
ness (Hendrycks et al., 2019) and may enhance the model performance, especially on small datasets.

Following this paradigm, modern object detectors, like Faster RCNN (Ren et al., 2016) and
YOLO (Redmon et al., 2016), generally add a lightweight task-speciﬁc layers on top of ImageNet
pre-trained backbones, so that the model can enjoy the aforementioned beneﬁts of pre-training.
However, the recently developed detection transformers1, e.g., DETR (Carion et al., 2020) and De-
formable DETR (Zhu et al., 2020), only pre-trains its CNN backbone, while leaving the core module,
a 12-layer transformer (including both encoder layers and decoder layers), trained from scratch in
downstream tasks. As a result, the model suffers from limited robustness against common corrup-
tions and the reliance on a large amount of training data for ﬁne-tuning. Though UP-DETR (Dai
et al., 2020) attempts to mitigate this problem by unsupervised pre-training, it requires an off-the-
shelf CNN backbone that has already be pre-trained. Moreover, the two-stage separate pre-training
methods may impair the model performance, since the two parts work together for object detection
during ﬁne-tuning.

The key reason that hinders existing detection transformers from beneﬁting from large-scale pre-
training can be attributed to the discrepancy between the upstream task and the downstream task.

∗This work was done during Wen Wang’s internship at JD Explore Academy. †Corresponding author.
1In this paper, we use detection transformer represents a class of object detectors built upon transformers,

while DETR represents the seminal work by (Carion et al., 2020).

1

Published as a conference paper at ICLR 2022

Firstly, the object detection-oriented model structures are difﬁcult to adapt to the ImageNet classiﬁ-
cation task. For example, the transformer decoder requires multiple query embeddings for detecting
objects, while for ImageNet classiﬁcation, there is only a single query embedding (class token) used.
If the decoder is included during pre-training, both the self-attention layers and the projections on
query embeddings in cross-attention layers may easily overﬁt to the single class token, making it
is hard to pre-train the decoder. Secondly, the upstream ImageNet classiﬁcation task misses some
crucial components for the downstream object detection task. For example, while the downstream
object detection task requires both localization and classiﬁcation for the objects of interest, the up-
stream classiﬁcation task only focuses on the latter. Moreover, the object relation modeling in object
detection, which is important for removing heuristic post-processing like non-maximum suppression
(NMS) (Carion et al., 2020), is absent in image classiﬁcation.

To mitigate these problems, we propose FP-DETR, a novel method that reformulates the pre-training
and ﬁne-tuning phases for detection transformers. It fully pre-trains a detection transformer on Im-
ageNet classiﬁcation task and smoothly ﬁne-tune it for object detection task though a task adaptor.
Concretely, during pre-training, we introduce an encoder-only transformer structure by removing
the decoder that can hardly be well pre-trained on the ImageNet classiﬁcation task. Moreover, since
both the CNN backbone and the transformer encoder in detection transformers can be seen as feature
extractor, we replace the complex CNN backbone with a simple multi-scale tokenizer and only uses
the transformer encoder for feature extraction. The resultant architecture is an efﬁcient transformer
encoder-only model. During ﬁne-tuning on the object detection task, we take inspiration from the
success of textual prompt (Liu et al., 2021a) in NLP and treat the query positional embeddings as
visual prompts to help the model attend to the target areas (prompting) and recognize the object. Our
key intuition is that if the model knows where to look at, it simply needs to identify the category of
the object in hand, as did during pre-training. Based on this motivation, we devise a lightweight task
adaptor to enhance the prompt ability of query positional embeddings and hence bridge the gap be-
tween upstream classiﬁcation task and downstream object detection task. It leverages self-attention
to emphasize the relation between query embeddings, such that the inter-object relationships, which
have been neglected during pre-training, can be captured during ﬁne-tuning.

Experiments show that the proposed FP-DETR achieves competitive performance on the challeng-
ing COCO dataset (Lin et al., 2014), and a better trade-off between the number of parameters and
detection accuracy. Moreover, FP-DETR is more robust against common corruptions and can gen-
eralize well to small-size datasets like Cityscapes (Cordts et al., 2016), attributing to the effective
fully pre-training.

2 RELATED WORK

Object Detection. Object detection is one of the fundamental tasks in computer vision (CV). Rep-
resentative object detection methods can be roughly categorized as the one-stage object detector,
e.g., YOLO (Redmon et al., 2016), SSD (Liu et al., 2016), and two-stage object detectors like
Faster-RCNN (Ren et al., 2016). While signiﬁcant progress has been made, these methods are not
end-to-end and rely on heuristic components like non-max suppression (NMS) and rule-based label
assignments. Recently, DETR (Carion et al., 2020) formulates the object detection task as a set
prediction problem and proposes an end-to-end pipeline to exploit transformer and bipartite match-
ing for detection. The success of DETR brought the recent surge of detection transformers. For
example, Deformable DETR (Zhu et al., 2020) proposes deformable attention to signiﬁcantly accel-
erate the model convergence and reduce the computational cost, while allowing the model to beneﬁt
from multi-scale features. Conditional DETR (Meng et al., 2021) narrows down the spatial range
for localizing the object regions via learning the decoder embedding conditioned on a spatial query.
REGO-DETR (Chen et al., 2021) mitigate the training difﬁculty through Region-of-Interest (RoI)
based detection reﬁnement. While effective, the crucial component in these models, i.e., a 12-layer
transformer, is trained from scratch, which limits the robustness and generalization ability of the
detection models.

Pre-training for Object Detection. The seminal work RCNN (Girshick et al., 2014) demonstrates
the many beneﬁts of ImageNet pre-training for object detection, e.g., better detection performance
and faster convergence. Since then, pre-training and ﬁne-tuning has become a well-established
paradigm for object detection. Recently, He et.al challenge the common belief by showing that

2

Published as a conference paper at ICLR 2022

Pre-training

Fine-tuning

Figure 1: Illustration of pre-training and ﬁne-tuning stages of FP-DETR.

training from scratch on datasets like COCO with a longer schedule can also produce a competitive
detection performance (He et al., 2019). However, Hendrycks et.al show that though comparable
in-domain performance can be achieved, training from scratch results in much worse out-of-domain
generalization (Hendrycks et al., 2019).
Inspired by the success of pre-training for CNN-based
object detectors, UP-DETR (Dai et al., 2020) proposes a novel random query patch detection pre-
training task to improve DETR’s performance and convergence. Though progress has been made,
their method relies on a pre-trained off-the-shelf CNN backbone. Moreover, the improvement is
limited, since the CNN backbone and the transformer are pre-trained separately while they need to
work jointly in the downstream task. Probably the most similar work to ours is YOLOS (Fang et al.,
2021), which also pre-trains and ﬁne-tunes an encoder-only transformer for object detection. How-
ever, it aims at demonstrating the feasibility of detecting objects using BERT (Devlin et al., 2018),
with minimum inductive bias. As a result, the discrepancy between the upstream and downstream
tasks, which signiﬁcantly degenerates the model performance, is not considered in YOLOS. By con-
trast, we aim at maximizing the beneﬁts of pre-training for the downstream object detection task.
To this end, we treat query positional embeddings as visual prompts and devise a novel task adaptor
to smoothly ﬁne-tune the pre-trained model on downstream tasks. Moreover, since object detec-
tion task often requires high-resolution images as input, inductive bias like sparsity is introduced to
reduce the computational costs.

Prompt-based Learning in NLP. Large-scale pre-trained models like GPT-3 (Brown et al., 2020)
advance the ﬁeld of NLP. However, such large-scale models are not designed for ﬁne-tuning. To bet-
ter utilize these models for downstream tasks, the “pre-train, prompt, and predict” paradigm emerges
in the NLP community, and has drawn increasing attention (Liu et al., 2021a). The new paradigm
reformulates the downstream task to mimic the task during pre-training with the help of a textual
prompt, to bridge the gap between upstream and downstream tasks. Speciﬁcally, pioneer work intro-
duces manually designed prompts to improve the ﬁne-tuning (Schick & Sch¨utze, 2020), or searches
the textual prompt for downstream tasks (Gao et al., 2020; Jiang et al., 2020). Though improvements
have been made, these prompts correspond to natural language phrases in discrete space can be sub-
optimal, since the neural networks are inherently continuous. To tackle this problem, P-tuning (Liu
et al., 2021b) proposes to optimize contiguous prompt embedding to better close the gap between
pre-training and ﬁne-tuning. In this paper, instead of reformulating the downstream task, we pro-
pose a new perspective and treat the query positional embeddings in detection transformers as visual
prompts. To this end, a task adaptor is devised to facilitate ﬁne-tuning on the downstream task.

3 METHOD

In this section, we ﬁrst revisit the preliminaries on detection transformers, then introduce the pre-
training and ﬁne-tuning phases of our method as shown in Figure 1. Through exploiting the large-
scale pre-training on ImageNet, our detection transformer can enjoy better detection performance,
improved robustness, and enhanced generalization.

3

Pre-trained Deformable Transformer Encoder LayerVisual PromptsPrompt EncoderPrediction    elephantperson× L Deformable Transformer Encoder LayerFlattened Multi-scale PatchesClass TokenPredictionCLSelephant× LPat#NPat#3Pat#2Pat#1CLSCLS ProjectionFlattened Multi-scale PatchesPat#NPat#3Pat#2Pat#1ProjectionLoc#KLoc#3Loc#2Loc#1Published as a conference paper at ICLR 2022

3.1 PRELIMINARIES ON DETECTION TRANSFORMERS

Detection transformers generally consist of the following parts: a backbone, a transformer encoder,
a transformer decoder, and a task-speciﬁc feed-forward network (FFN). The backbone F can be
decomposed as multiple stages, i.e., F = f l ◦ · · · ◦ f 2 ◦ f 1, where each stage fi takes the output
from the previous stage as input and outputs the down-sampled feature map at stage i. For an
input image I ∈ RH×W ×3, the backbone extracts multi-scale features, which are ﬂattened and
projected to 1-D sequence representation. Afterwards, the sequence representation is added with
position and level embedding to obtained x ∈ RN ×D, where N = (cid:80)L
l is the sequence
length, Sl is the down-sampling rate at l-th feature level, and D is the dimension of the feature
embedding. The transformer general consists of both encoder and decoder, and takes the sequence
representation extracted from the image and the query embeddings as input for context modeling.
The ﬁnal prediction is made by applying FFN to the object representations.

l=1 HW/S2

While Deformable DETR (Zhu et al., 2020) contains four different feature levels with different
down-sampling rates, most other detection transformers, including DETR (Carion et al., 2020),
Conditional DETR (Meng et al., 2021), contains only a single feature level.

3.2 PRE-TRAINING

Encoder-only Transformer. As described in Section 1, for image classiﬁcation, only a single class
token is required for context aggregation. The six decoder layers trained on only one class token is
prone to overﬁtting. To tackle this problem, we remove the decoder during pre-training on the image
classiﬁcation task, which leads to an encoder-only transformer structure. We follow Deformable
DETR (Zhu et al., 2020) to adopt multi-scale deformable attention with four different feature levels
to build our model. It largely alleviates the slow convergence of DETR (Carion et al., 2020), with
the help of inductive bias like sparsity.

To perform image classiﬁcation, an additional class token added with corresponding positional em-
bedding xcls ∈ RD is concatenated with the sequence feature, for aggregating global context from
the image tokens. Thus the input to the encoder-only transformer z0 can be written as,

z0 = (cid:2)xcls; x1; x2; · · · ; xN (cid:3) .
(1)
Afterward, the input sequence feature is processed by T different transformer encoder layers, each
consists of a multi-scale deformable self-attention (MSDSA) layer and an MLP layer. Residual
connections and layer norm (LN) are applied after each sub-layer, i.e.,
z(cid:48)
t = LN(MSDSA(zt−1) + zt−1)
zt = LN(MLP(z(cid:48)

(3)
Final predictions are made on class token at the T -th layer, i.e., z0
T . During implementation, the
class token is treated as an additional feature level, with only one resolution. Since multi-scale
deformable attention assumes different levels of feature maps are spatially aligned, which does not
hold for the class token in xcls, attention mask is utilized to remove the information ﬂow from xcls.

t = 1 . . . T.

t = 1 . . . T,

t) + z(cid:48)

(2)

t)

Lightweight Multi-scale Tokenizer. The transformer encoder’s ability to serve as a general feature
extractor has been proved by recent advances in vision transformers (Dosovitskiy et al., 2020).
While existing detection transformers contain both sophisticated backbone and transformer encoder
for feature extraction, we argue that such design brings unnecessary complexity to the detection
model. Instead, we replace the complex backbone with simple convolutional layers that work as
a multi-scale tokenizer. Speciﬁcally, the feature extractor at each stage fi is a simple single-layer
convolutional layer that is designed only for down-sampling the feature map to expected resolution.
Such design signiﬁcantly simpliﬁes the model and makes it easy to pre-train the detection model
on ImageNet classiﬁcation task. The detailed structure of our multi-scale tokenizer is shown in
Appendix A.1.

Discussion. Most of the existing vision transformers adopt an encoder-only structure (Dosovitskiy
et al., 2020; Liu et al., 2021c; Xu et al., 2021). By reformulating the detection transformer to an
encoder-only structure, we solve the problem that the decoder is difﬁcult to pre-train and simplify
the model structure. Moreover, it allows us to take advantage of the existing experience in training
vision transformers.
In Appendix A.2, we explore pre-training on ImageNet using the encoder-
decoder structure, however, the performance is much worse than that of the encoder-only structure.

4

Published as a conference paper at ICLR 2022

Textual Prompt

Visual Prompt

Figure 2: Analogy between textual prompt in NLP and visual prompt in CV.

3.3 FINE-TUNING

During ﬁne-tuning, the single class token is replaced by Nq query content embeddings (Carion
et al., 2020) xobj ∈ RNq×D for object detection. The content embeddings are also added with query
positional embeddings (Carion et al., 2020) p ∈ RNq×D before feed into the transformer encoder.
Thus the input to the transformer encoder is:

(cid:104)

z0 =

obj + p1; x2
x1

obj + p2; · · · ; xNq

obj + pNq ; x1; x2; · · · ; xN (cid:105)

.

(4)

The class token for image classiﬁcation and the query content embeddings for object detection are
both designed for context aggregation. While the former focuses on global context aggregation, the
latter focus on aggregating the local context for one speciﬁc object instance.

Query Positional Embeddings as Visual Prompts. The textual prompt in NLP is shown in Fig-
ure 2. It reformulates the downstream task to mimic the pre-training one, so that the pre-trained
model can better handle the downstream task. However, ﬁnding the optimal textual prompt is not
easy, recent work such as P-tuning (Liu et al., 2021b) searches prompts in a contiguous space to
bridge the gap between the upstream and downstream tasks.

Object detection can be decomposed into two subtasks, i.e., localization and classiﬁcation. Our intu-
ition is that if the classiﬁer pre-trained on ImageNet knows where to look at, it can easily recognize
the object within the speciﬁed region, as did during ImageNet classiﬁcation pre-training. In our de-
tection transformer, the query position embeddings are mapped to the reference points, which guide
different query content embeddings to sample the corresponding context from the image content.
In other words, the query position embedding works as a visual cue to point out the image region
that the model should focus on, as shown in ﬁgure 2 (b). From this perspective, the query position
embedding serves as a visual prompt, which is analogous to the textual prompt in NLP. Further,
the process of training the model to localize objects corresponds to the process of searching textual
prompts in contiguous space. And the process of ﬁnal classiﬁcation corresponds to the process of
ﬁlling in the blanks and answers mapping in NLP (Liu et al., 2021a).

Improving Fine-tuning with Task Adaptor. Based on the above motivation, we propose a task
adaptor to improve the query positional embeddings’ ability to prompt. Speciﬁcally, the inter-object
relationship can help the model better identify object regions. However, this crucial part is missing
during ImageNet classiﬁcation pre-training, since only one class token is used. To this end, we
devise the task adapter to improve the modeling of the inter-object relationships. It pre-processes
the visual prompts before they are sent to each pre-trained encoder layer, i.e.,

z0:Nq
t

= TaskAdaptor

(cid:16)

z0:Nq
t

(cid:17)

t = 1 . . . T.

(5)

By default, the task adaptor is instantiated as a self-attention layer. Afterward, the sequence is pro-
cessed by the transformer encoder layers, as did in Equation 2 and 3. The object pattern embedding
in the last encoder layer, i.e., z0:Nq
is used for ﬁnal prediction. Our task adaptor helps the visual

t

5

Visual Transformer Pre-trained on Image Classification TaskI love this movie. Overall, it is a good movie.Language Model Pre-trained on Masked Language Modeling Task I love this movie. Overall, it is a [mask] movie.PositivePerson Textual Prompt (what to fill in)Visual Prompt (where to attend)Published as a conference paper at ICLR 2022

Model Layers Hidden size D MLP size Heads Params Acc@Top-1
1024
Lite
1536
Small
Base
1920

11M
23M
35M

77.3
80.8
82.0

256
384
480

8
8
10

12
12
12

Table 1: Details of our FP-DETR variants. Parameters (Params) and top-1 accuracy (Acc@Top-1)
here are calculated on the upstream ImageNet-1k (Deng et al., 2009) classiﬁcation task.

prompts for individual objects to make associations with other instances, making them more suitable
for the downstream object detection task.

Discussion. Recently, Shen et al. (2021) ﬁnd the connection between named entity recognition
(NER) in NLP and object detection in CV, and propose a two-stage method for NER. Similarly, We
propose a new perspective to view query positional embeddings as visual prompts. However, the
existing pre-trained visual models are not as powerful as pre-trained language models (Brown et al.,
2020). As a result, we are unable to ﬁx the pre-trained models and only tune the visual prompts, to
perform few-shot or even zero-shot learning on downstream tasks, as did in NLP. But still, this new
perspective helps us to better understand the detection transformers and the discrepancy between
the pre-training and downstream tasks, which further helps us design the task adaptor to smoothly
ﬁne-tune the pre-trained model on downstream tasks.

4 EXPERIMENTS

4.1

IMPLEMENTATION DETAILS

Datasets. Following the common practice, our detector is pre-trained on ImageNet (Deng et al.,
2009) and ﬁne-tuned on COCO 2017 (Lin et al., 2014) train set. Evaluation results on the val set
of COCO 2017 are reported. To evaluate the models’ robustness against common corruptions, we
report the model performances on COCO-C (Michaelis et al., 2019), which is obtained by applying
corruption synthesis algorithm to COCO. Besides, we evaluated the model’s generalization ability
by ﬁne-tuning on the small-size dataset, i.e., Cityscapes dataset (Cordts et al., 2016).

Model Variants The lite version of our model follows the hyper-parameters of the Deformable
Transformer in Deformable DETR (Zhu et al., 2020). Speciﬁcally, the encoder-only transformer
contains 12 self-attention layers, each layer with 8 heads, the dimension of each head is 32, and the
dimension of the MLP layer in FFN is 1024. Furthermore, we introduce two other model variants by
changing the number of heads, or dimensions per head, while keeping the model depth unchanged.
Table 1 summarizes the different variants of our model.

Training & Fine-tuning. By default, our FP-DETR is pre-trained on ImageNet-1k (Deng et al.,
2009) for 300 epochs with AdamW (Loshchilov & Hutter, 2018) optimizer and cosine learning rate
scheduler. Training strategies in DeiT (Touvron et al., 2021a) are adopted, and the image size is set
as 224×224. We use a batch size of 1,024 for training, and the initial learning rate is set as 5×10−4.
After pre-training, models are ﬁne-tuned for 50 epochs with AdamW optimizer on the downstream
tasks. The learning rate is initialized as 1 × 10−4 and decreased by a factor of 0.1 at the 40th epoch.
We follow the implementation of Deformable DETR to apply deep supervision (Lee et al., 2015) on
the last six encoder layers. The class token and the query content embedding are concatenated with
the image tokens at an intermediate (the 7th) encoder layer, following CaiT (Touvron et al., 2021b).
Besides, we set both the number of sampling points and the feature levels in multi-scale deformable
attention as 4, and the number of object query embeddings as 300. Models are ﬁne-tuned with a
batch size of 32. All experiments are implemented on the NVIDIA A100 GPU.

4.2 COMPARISON WITH STATE-OF-THE-ARTS

As can be seen in Table 2, FP-DETR achieves competitive performance on COCO 2017 val set.
Speciﬁcally, FP-DETR-Base is comparable with state-of-the-art Deformable DETR, while its small
variant with 24M parameters can match the performance of UP-DETR and outperforms Conditional
DETR and DETR with about 40M parameters. It performs better on detecting small objects and has
better localization ability.

6

Published as a conference paper at ICLR 2022

Table 2: Comparision of FP-DETR with other detection transformers on COCO 2017 val set. Mod-
els are categorized as encoder-only (Enc) and encoder-decoder (Enc-Dec) according to the trans-
former structure. † indicates the model is pre-trained on ImageNet-21k.

Structure Backbone Epochs AP AP50 AP75 APS APM APL Params
Method
41M
Enc-Dec ResNet-50
DETR
42.0 62.4 44.2 20.5 45.8 61.1
UP-DETR
41M
Enc-Dec ResNet-50
42.8 63.0 45.3 20.8 47.1 61.7
44M
Conditional DETR Enc-Dec ResNet-50
40.9 61.8 43.3 20.8 44.6 59.2
Deformable DETR Enc-Dec ResNet-18
23M
40.1 58.4 43.7 22.0 43.4 53.0
33M
Deformable DETR Enc-Dec ResNet-34
42.3 60.7 46.0 24.2 45.8 56.1
Deformable DETR Enc-Dec ResNet-50
40M
43.8 62.6 47.7 26.4 47.1 58.0
36.1 56.5 37.1 15.3 38.5 56.2
YOLOS-S
31M
YOLOS-B
42.0 62.2 44.5 19.5 45.3 62.1 127M
11M
37.2 56.5 40.4 21.7 40.0 48.6
FP-DETR-Lite
24M
42.5 62.6 45.9 25.3 45.5 56.9
FP-DETR-Small
FP-DETR-Base
36M
43.3 63.9 47.7 27.5 46.1 57.0
FP-DETR-Base†
36M
43.7 64.1 47.8 26.5 46.7 58.2

500
300
50
50
50
50
150
150
50
50
50
50

Enc
Enc
Enc
Enc
Enc
Enc

-
-
-
-
-
-

Moreover, FP-DETR shows a better trade-off between model parameters and detection accuracy
compared to Deformable DETR. For example, FP-DETR-Small outperforms Deformable DETR
with ResNet-18 (He et al., 2016) backbone, which has a similar amount of parameters.
It also
slightly outperforms Deformable DETR with ResNet-34 backbone, which has about 10M more
parameters. The smallest version of our model, FP-DETR-lite, achieves 37.2 mAP on COCO 2017
with only 11M parameters.

Besides, FP-DETR signiﬁcantly outperforms YOLOS-Base which also adopts a transformer
encoder-only structure, in terms of both model efﬁciency and detection accuracy. This is mainly
due to the inductive bias introduced in our model as Deformable DETR and the prompt-inspired
task adaptor for more effective ﬁne-tuning, which better bridges the gap between pre-training and
ﬁne-tuning.

The last row shows the result of our FP-DETR-Base ﬁne-tuned from ImageNet-21 pre-training, de-
noted as FP-DETR-Base†. As can be seen, FP-DETR can beneﬁt from pre-training at a larger scale,
and FP-DETR-Base† with 36M parameters matches the performance of Deformable DETR with
40M parameters. In Appendix A.3, we also show the result of Deformable DETR with ResNet-50
backbone pre-trained on ImageNet-21k. Interestingly, the result is slightly worse than Deformable
DETR with ImageNet-1k pre-trained backbone. We conjuncture that only pre-training part of the
model makes it hard for the randomly initialized transformer adapts to the pre-trained CNN back-
bone, especially when the backbone has already been well-trained.

Table 3: Ablations on the task adaptor on COCO 2017 val set.

N/A

Task Adaptor Shared AP AP50 AP75 APS APM APL Params
11M
11M
11M
12M
11M

w/o
Bi-LSTM
Bi-LSTM
Self-attention
Self-attention

32.8
34.6
34.9
36.8
37.2

18.2
19.1
20.2
20.9
21.7

35.8
38.1
38.0
39.8
40.0

43.6
45.4
46.9
49.1
48.6

50.9
52.9
53.0
56.0
56.5

35.2
37.3
37.5
39.5
40.4

(cid:88)

(cid:88)

4.3 ABLATIONS ON THE TASK ADAPTOR

To gain a better understanding of the effectiveness of the proposed task adaptor, we perform ablation
studies on the lite version of FP-DETR, as shown in Table 3. We have the following observations.
First, removing the task adaptor results in signiﬁcant drops in model performance. Without the
task adaptor, the model can not well capture the inter-object relationships, which is crucial for re-
moving duplicates and improving object recognition. Second, the task adaptor instantiated with
a bidirectional-LSTM layer also helps to adapt the pre-trained model to the downstream task, but
it works slightly worse than a self-attention layer. We conjuncture that the bidirectional-LSTM is

7

Published as a conference paper at ICLR 2022

slightly worse on modeling long-range dependency compared to the self-attention layer, especially
when the number of query embeddings reaches 300. By default, we adopt the self-attention layer
for the task adaptor. In this way, our FP-DETR is a pure transformer encoder structure. Third, in-
terestingly, a single task adaptor shared by visual prompts from different transformer encoder layers
performs slightly better than specialized task adaptors for each layer. Since the task adaptor is trained
from scratch on the downstream task, we conjuncture that the shared task adaptor sees diverse data
from different levels compared to the non-shared ones, and is thus more sufﬁciently trained. These
experiments validate the effectiveness of our task adaptor on modeling object relationships, which
is crucial for object detection.

Table 4: Comparison of model robustness on the COCO-C dataset.

Method

Mean

DETR
UP-DETR
Conditional DETR
Deformable-DETR
YOLOS-S
FP-DETR-lite
FP-DETR-Small
FP-DETR-Base

Method

DETR
UP-DETR
Conditional DETR
Deformable-DETR
YOLOS-S
FP-DETR-Lite
FP-DETR-Small
FP-DETR-Base

19.1
21.6
20.3
20.7
18.9
18.9
22.8
23.7

Noise
Gauss Shot
16.5
16.8
22.2
22.1
18.6
18.6
19.6
19.8
15.1
15.3
17.0
17.0
20.3
20.3
22.7
22.5

Weather

Blur

Impul Defocus Glass Motion Zoom
13.8
12.7
13.4
18.7
14.8
16.0
13.8
16.4
15.7
14.0
15.1
15.2
18.3
18.1
18.9
20.4

20.0
20.1
20.5
20.1
19.4
18.6
22.3
22.9

17.7
18.3
18.4
17.9
20.2
17.5
21.2
21.7

6.5
6.9
7.1
6.6
7.1
6.5
8.0
8.0

Digital

Snow Frost
21.4
16.4
24.2
19.3
22.9
18.2
23.6
18.3
21.2
17.7
22.7
18.2
27.1
22.9
27.1
22.4

Fog Bright Contrast Elastic
23.2
29.0
23.0
31.6
24.0
30.0
24.0
31.7
22.8
26.5
21.3
28.4
25.5
33.1
26.1
33.4

18.8
21.4
20.6
21.2
19.7
20.5
25.0
25.3

35.0
36.9
34.7
37.0
29.5
32.5
37.3
38.0

Pixel
18.3
20.7
19.4
19.3
17.1
14.8
20.2
20.0

JPEG
19.8
24.6
20.5
21.0
21.8
18.2
23.1
25.6

Table 5: Comparison of model generalization on the small-size Cityscapes dataset.

Structure Epochs
Method
Enc-Dec
DETR
Enc-Dec
UP-DETR
Conditional DETR Enc-Dec
Deformable DETR Enc-Dec
YOLOS-S
FP-DETR-Lite
FP-DETR-Small
FP-DETR-Base

500
300
150
50
150
50
50
50

Enc
Enc
Enc
Enc

AP
15.9±0.9
23.8±1.3
21.1±0.7
27.3±0.6
9.8±0.1
26.7±0.6
28.6±0.2
29.6±0.5

AP50
34.8±1.6
45.7±2.2
42.7±1.3
49.2±0.6
25.3±0.4
49.1±0.8
52.0±0.6
53.6±0.9

AP75
12.7±1.0
20.8±1.2
18.8±1.6
26.3±0.8
6.1±0.4
25.7±0.7
26.9±0.8
28.4±0.4

APS
2.9±0.1
4.0±0.5
3.6±0.3
8.7±0.2
1.9±0.2
9.7±0.7
10.1±0.6
11.2±0.7

APM
13.5±0.7
20.3±1.9
19.8±0.3
28.2±0.8
8.1±0.4
28.6±0.6
30.2±0.6
30.9±0.8

APL
33.8±2.1
46.6±1.9
41.1±1.0
45.7±0.7
20.7±0.4
42.8±0.7
46.3±0.7
47.4±1.2

Params
41M
41M
44M
40M
31M
11M
24M
36M

4.4 ROBUSTNESS TO COMMON CORRUPTIONS

Model robustness (He & Tao, 2020) is critical to trustworthy AI applications like autonomous driv-
ing. To this end, we evaluate the object detectors’ robustness against common corruptions on COCO-
C (Michaelis et al., 2019). As shown in Table 4, all detectors suffer signiﬁcant performance drops
under this rigorous condition. However, FP-DETR suffers the least performance drop compared to
existing detection transformers. Notably, FP-DETR-Base performs best on 14 out of 15 types of
corruptions, though it has comparable performance to Deformable DETR on the clean COCO 2017
val set. This is because the pre-training on ImageNet helps FP-DETR learn more generalizable
representation. This is coherent with the observation in Hendrycks et al. (2019).

The ﬁrst two rows in Figure 3 provide some qualitative results of our FP-DETR-Base on the COCO-
C dataset. As can be seen, the images are signiﬁcantly degenerated due to the existence of various
corruptions, like elastic transform, zoom blur, fog, etc. However, FP-DETR-Base still manages to
produce plausible results under low visibility and large distortion, which manifests its robustness.

8

Published as a conference paper at ICLR 2022

Figure 3: Qualitative detection results of FP-DETR on the COCO-C and Cityscapes datasets are
shown in the ﬁrst two rows and the last row, respectively.

4.5 GENERALIZATION TO SMALL-SIZE DATASET

In real-world applications, collecting a large amount of data is often infeasible. As a result, the
models are required to perform well by training on only limited data. Table 5 shows the result of
models ﬁne-tuned on the Cityscapes dataset with only 2,975 training images. All models are trained
with a batch size of 8 to guarantee enough training iterations. The results are averaged over ﬁve
repeated runs with different random seeds. As can be seen, most detection transformers, including
DETR, UP-DETR, Conditional DETR, and YOLOS-B, perform poorly under this condition. This
is partly due to the lack of inductive bias in those models. As a result, the model requires a large
amount of data to learn the sparsity that is beneﬁcial for object detection on images. Deformable
DETR performs better, thanks to the sparsity inductive bias introduced in the deformable trans-
former. However, it still performs worse than our FP-DETR, since the transformer in Deformable
DETR is trained from scratch. Speciﬁcally, both FP-DETR-Base and FP-DETR-Small outperform
Deformable DETR with more parameters. Notably, our FP-DETR-Lite with only 11M parameters
matches the performance of 40M Deformable DETR, attributing to the end-to-end fully pre-training
of the transformer.

The last row in Figure 3 shows some qualitative results of our FP-DETR-Base on the Cityscapes
dataset. As can be seen, FP-DETR quickly adapts to the small-size dataset, and produces accurate
object detection results, even for the small cars in the distant area. These results demonstrate the
generalization ability of our method.

5 CONCLUSION

In this paper, we propose a novel detection transformer named FP-DETR that can take advantage of
pre-training on upstream datasets to enable better robustness and generalization. FP-DETR contains
a simple encoder-only structure for fully pre-training and can be smoothly ﬁne-tuned for object
detection via a task adapter. It leverages query positional embeddings as visual prompts to help
the model attend to the target area (prompting) and recognize the object, effectively mitigating the
gap between the upstream ImageNet classiﬁcation task and the downstream object detection task.
Experiments show that FP-DETR not only achieves competitive performance on the challenging
COCO dataset, but also gains better robustness against common corruptions and generalization to
small-size datasets.

9

         Published as a conference paper at ICLR 2022

ACKNOWLEDGEMENT

This work is supported by National Key R&D Program of China under Grant 2020AAA0105701,
National Natural Science Foundation of China (NSFC) under Grants 61872327, and Major Special
Science and Technology Project of Anhui (No. 012223665049). Dr Jing Zhang is supported by
ARC FL-170100117.

ETHICS STATEMENT

We acknowledge that all co-authors of this work have read and commit to adhering to the ICLR
Code of Ethics.

REPRODUCIBILITY STATEMENT

The authors strive to ensure the reproducibility of the experimental results. Speciﬁcally, implemen-
tation details are provided in Section 4.1, Section A.1, and Section A.3. The related studies with
the same experimental setup are carefully cited and referred to. Moreover, the code will be made
publicly available to ensure reproducibility.

REFERENCES

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and
In European Conference

Sergey Zagoruyko. End-to-end object detection with transformers.
on Computer Vision, pp. 213–229. Springer, 2020.

Zhe Chen, Jing Zhang, and Dacheng Tao. Recurrent glimpse-based decoder for detection with

transformer. arXiv preprint arXiv:2112.04632, 2021.

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo
Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban
In Proceedings of the IEEE conference on computer vision and pattern
scene understanding.
recognition, pp. 3213–3223, 2016.

Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated
data augmentation with a reduced search space. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition Workshops, pp. 702–703, 2020.

Zhigang Dai, Bolun Cai, Yugeng Lin, and Junying Chen. Up-detr: Unsupervised pre-training for
object detection with transformers. In Proceedings of the IEEE conference on computer vision
and pattern recognition, 2020.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hi-
erarchical image database. In 2009 IEEE conference on computer vision and pattern recognition,
pp. 248–255. IEEE, 2009.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Terrance DeVries and Graham W Taylor. Improved regularization of convolutional neural networks

with cutout. arXiv preprint arXiv:1708.04552, 2017.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An
image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint
arXiv:2010.11929, 2020.

10

Published as a conference paper at ICLR 2022

Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, and
Wenyu Liu. You only look at one sequence: Rethinking transformer in vision through object
detection. arXiv preprint arXiv:2106.00666, 2021.

Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot

learners. arXiv preprint arXiv:2012.15723, 2020.

Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for ac-
curate object detection and semantic segmentation. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pp. 580–587, 2014.

Fengxiang He and Dacheng Tao. Recent advances in deep learning theory.

arXiv preprint

arXiv:2012.10931, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-
nition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp.
770–778, 2016.

Kaiming He, Ross Girshick, and Piotr Doll´ar. Rethinking imagenet pre-training. In Proceedings of

the IEEE/CVF International Conference on Computer Vision, pp. 4918–4927, 2019.

Dan Hendrycks, Kimin Lee, and Mantas Mazeika. Using pre-training can improve model robustness
and uncertainty. In International Conference on Machine Learning, pp. 2712–2721. PMLR, 2019.

Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. How can we know what language
models know? Transactions of the Association for Computational Linguistics, 8:423–438, 2020.

Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, and Zhuowen Tu. Deeply-

supervised nets. In Artiﬁcial intelligence and statistics, pp. 562–570. PMLR, 2015.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-
train, prompt, and predict: A systematic survey of prompting methods in natural language pro-
cessing. arXiv preprint arXiv:2107.13586, 2021a.

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and
Alexander C Berg. Ssd: Single shot multibox detector. In European conference on computer
vision, pp. 21–37. Springer, 2016.

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt

understands, too. arXiv preprint arXiv:2103.10385, 2021b.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining
Guo. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint
arXiv:2103.14030, 2021c.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Confer-

ence on Learning Representations, 2018.

Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, and Jing-
dong Wang. Conditional detr for fast training convergence. In Proceedings of the IEEE interna-
tional conference on computer vision, 2021.

Claudio Michaelis, Benjamin Mitzkus, Robert Geirhos, Evgenia Rusak, Oliver Bringmann, Alexan-
der S. Ecker, Matthias Bethge, and Wieland Brendel. Benchmarking robustness in object detec-
tion: Autonomous driving when winter is coming. arXiv preprint arXiv:1907.07484, 2019.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Uniﬁed,
real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 779–788, 2016.

11

Published as a conference paper at ICLR 2022

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: towards real-time object
IEEE transactions on pattern analysis and machine

detection with region proposal networks.
intelligence, 39(6):1137–1149, 2016.

Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for

the masses. arXiv preprint arXiv:2104.10972, 2021.

Timo Schick and Hinrich Sch¨utze. It’s not just size that matters: Small language models are also

few-shot learners. arXiv preprint arXiv:2009.07118, 2020.

Yongliang Shen, Xinyin Ma, Zeqi Tan, Shuai Zhang, Wen Wang, and Weiming Lu. Locate and

label: A two-stage identiﬁer for nested named entity recognition. In ACL, 2021.

Leslie N Smith. A disciplined approach to neural network hyper-parameters: Part 1–learning rate,

batch size, momentum, and weight decay. arXiv preprint arXiv:1803.09820, 2018.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethink-
In Proceedings of the IEEE conference on

ing the inception architecture for computer vision.
computer vision and pattern recognition, pp. 2818–2826, 2016.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and
Herv´e J´egou. Training data-efﬁcient image transformers & distillation through attention.
In
International Conference on Machine Learning, pp. 10347–10357. PMLR, 2021a.

Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Herv´e J´egou. Going
deeper with image transformers. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pp. 32–42, 2021b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
In Conference on Neural In-

Lukasz Kaiser, and Illia Polosukhin. Attention is all you need.
formation Processing Systems, 2017.

Yufei Xu, Qiming Zhang, Jing Zhang, and Dacheng Tao. Vitae: Vision transformer advanced by
exploring intrinsic inductive bias. Advances in Neural Information Processing Systems, 34, 2021.

Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: De-
formable transformers for end-to-end object detection. In International Conference on Learning
and Representations, 2020.

12

Published as a conference paper at ICLR 2022

A APPENDIX

A.1 STRUCTURE OF THE MULTI-SCALE TOKENIZER

Our lightweight multi-scale tokenizer extracts feature maps of 4 different levels to construct the
input sequences to the encoder-only transformer. Different from existing detection transformers,
which use a sophisticated ResNet-50 backbone for feature extraction, our multi-scale tokenizer
simply downsamples the input image to desired feature resolution. Speciﬁcally, we perform non-
overlapping patch embedding, similar to the seminal ViT (Dosovitskiy et al., 2020). The only differ-
ence is that we perform multi-scale tokenization and adopt a hierarchical structure to progressively
embed the tokens of large resolution. The detailed structure of our multi-scale tokenizer is shown in
Table 6. The dimension D(cid:48) is 192, 384, and 384, respectively for our Lite, Small, and Base model.
After patch embedding, the 2D sequence is ﬂattened to 1D sequence and further projected to tokens
with dimension D.

Table 6: The architecture of the lightweight multi-scale tokenizer.

Multi-scale Tokenizer
Conv 8 × 8 × D(cid:48), stride 8, pad 0
LayerNorm
Conv 2 × 2 × D(cid:48), stride 2, pad 0
LayerNorm
Conv 2 × 2 × D(cid:48), stride 2, pad 0
LayerNorm
Conv 2 × 2 × D(cid:48), stride 2, pad 0
LayerNorm

A.2 EXPLORING ENCODER-DECODER STRUCTURE

While most detection transformers contain an encoder-decoder transformer, existing study on vi-
sion transformers extensively explores the encoder-only transformer for image classiﬁcation. This
results in a discrepancy between the upstream ImageNet classiﬁcation task and the downstream
object detection task. To mitigate this problem, we ﬁrst explore pre-training the encoder-decoder
transformer on the ImageNet classiﬁcation task, using existing pre-training technologies (Touvron
et al., 2021a). Speciﬁcally, we pre-train and ﬁne-tune an encoder-decoder transformer structure,
denoted as FP-EncDec, on the ImageNet-1k dataset, follow the same implementation in 4.1. The
model follows the design of our FP-DETR-Lite, except the last 6 layers of self-attention layers are
replaced by 6 decoder layers and the task adaptor is removed. In each decoder layer, a standard
self-attention layer (Vaswani et al., 2017) and a multi-scale deformable attention layer (Zhu et al.,
2020) are applied sequentially, following Deformable DETR. Two variants of FP-EncDec are con-
sidered: (1) pre-training with a single class token, follow the common practice in training vision
transformers (Dosovitskiy et al., 2020); and (2) pre-training with 300 class tokens, which equals the
number of object query embeddings for ﬁne-tuning. The 300 class tokens are pooled into a single
class token before ﬁnal classiﬁcation.

Table 7: Comparison of encoder-decoder transformers and encoder-only transformer for pre-training
and ﬁne-tuning.

Method

Structure

FP-EncDec
FP-EncDec
FP-DETR-Lite

Enc-Dec
Enc-Dec
Enc

Token
Number
1
300
1

Acc@Top-1 AP AP50 AP75 APS APM APL Params

72.0
76.4
77.3

33.1 51.8 35.7 17.9 35.9 43.4 12M
35.2 54.0 37.9 20.9 38.1 47.0 12M
37.2 56.5 40.4 21.7 40.0 48.6 11M

The results for both pre-training on ImageNet-1k and ﬁne-tuning on COCO 2017 dataset are shown
in Table 7. As can be seen, pre-training the transformer decoder with a single class token performs
poorly on the ImageNet classiﬁcation task, since both the self-attention layers and the projections
on the class token in the cross-attention layer are trained on a single class token, which could easily

13

Published as a conference paper at ICLR 2022

lead to overﬁtting. The low pre-training accuracy also limits the model’s detection performance on
the downstream task. By contrast, FP-EncDec pre-trained with 300 class tokens performs better
on ImageNet-1k classiﬁcation, the top-1 accuracy is 4.4 higher than the single class token counter-
part, since the decoder is more sufﬁciently trained. However, it is still inferior to our encoder-only
FP-DETR-Lite. We conjuncture that this is caused by (1) existing training techniques for vision
transformers have been heavily tuned towards the encoder-only structure, which can be sub-optimal
for the encoder-decoder transformer; and (2) the discrepancy between upstream and downstream
tasks also degenerates the model’s performance on object detection, even if the decoder has been
pre-trained. We expect these ﬁndings may provide useful insights to the community to rethink the
current paradigm of pre-training vision transformers, and pay more attention to the pre-training of
encoder-decoder transformers.

A.3 PRE-TRAINING ON IMAGENET-21K

For ImageNet-21k pre-training, we follow the pipeline in Ridnik et al. (2021). The ImageNet-21k is
ﬁrst pre-processed by three steps: (1) cleaning invalid classes; (2) validation split; and (3) image re-
sizing. Afterward, the model is trained on the processed dataset (ImageNet-21k-P) using the seman-
tic softmax training (Ridnik et al., 2021). Speciﬁcally, our model is pre-trained on ImageNet-21k-P
with a batch size of 4096 for 80 epochs. The model is initialized from ImageNet-1K pre-trained
weights. The learning rate is initialized as 3e-4 and scheduled using one-cycle policy (Smith, 2018).
RandAugment (Cubuk et al., 2020), Cutout (DeVries & Taylor, 2017), Label-smoothing (Szegedy
et al., 2016), and True-weight-decay (Loshchilov & Hutter, 2018) are adopted for regularization.
For more details on pre-training, please refer to Ridnik et al. (2021).

For ﬁne-tuning FP-DETR further, we adopt the same implementation details in 4.1. For a fair
comparison, we also ﬁnetuned Deformable DETR (Zhu et al., 2020) with an ImageNet-21k-P pre-
trained ResNet-50 backbone. The ResNet-50 pre-trained weight is taken from the ofﬁcial release2
of Ridnik et al. (2021).

Table 8: Comparision of models ﬁne-tuned from ImageNet-1k and ImageNet-21k pre-trained
weights. † indicates the model is pre-trained on ImageNet-21k.

Method
Deformable DETR Enc-Dec R50
Deformable DETR† Enc-Dec R50
FP-DETR-Base
FP-DETR-Base†

Structure Bone Epochs AP AP50 AP75 APS APM APL params
40M
40M
36M
36M

43.8 62.6 47.7 26.4 47.1 58.0
42.9 62.1 46.6 25.1 46.3 57.7
43.3 63.9 47.7 27.5 46.1 57.0
43.7 64.1 47.8 26.5 46.7 58.2

50
50
50
50

Enc
Enc

-
-

The result is shown in Table 8. As can be seen, our FP-DETR can beneﬁt from fully pre-training
on the larger-scale ImageNet-21k dataset as well as ﬁne-tuning with the task adaptor. Counter-
intuitively, the performance of Deformable DETR degenerates when using the weights pre-trained
on ImageNet-21k. We conjuncture that only pre-training the CNN backbone of the model makes
it difﬁcult for the randomly initialized transformer to adapt to the pre-trained backbone, especially
when the backbone has already been well-trained. More research efforts should be made to further
dig into this problem.

2https://github.com/Alibaba-MIIL/ImageNet21K

14

