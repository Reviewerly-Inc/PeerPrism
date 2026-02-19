Under review as a conference paper at ICLR 2023

DATEFORMER: TRANSFORMER EXTENDS LOOK-BACK
HORIZON TO PREDICT LONGER-TERM TIME SERIES

Anonymous authors
Paper under double-blind review

ABSTRACT

Transformers have demonstrated impressive strength in long-term series forecast-
ing. Existing prediction research mostly focused on mapping past short sub-series
(lookback window) to future series (forecast window). The longer training dataset
time series will be discarded, once training is completed. Models can merely rely
on lookback window information for inference, which impedes models from ana-
lyzing time series from a global perspective. And these windows used by Trans-
formers are quite narrow because they must model each time-step therein. Under
this point-wise processing style, broadening windows will rapidly exhaust their
model capacity. This, for fine-grained time series, leads to a bottleneck in informa-
tion input and prediction output, which is mortal to long-term series forecasting.
To overcome the barrier, we propose a brand-new methodology to utilize Trans-
former for time series forecasting. Specifically, we split time series into patches
by day and reform point-wise to patch-wise processing, which considerably en-
hances the information input and output of Transformers. To further help models
leverage the whole training set’s global information during inference, we distill the
information, store it in time representations, and replace series with time represen-
tations as the main modeling entities. Our designed time-modeling Transformer—
Dateformer yields state-of-the-art accuracy on 7 real-world datasets with a 33.6%
relative improvement and extends the maximum forecast range to half-year.1

1

INTRODUCTION

Time series forecasting is a critical demand across many application domains, such as energy con-
sumption, economics planning, traffic and weather prediction. This task can be roughly summed
up as predicting future time series by observing their past values. In this paper, we study long-term
forecasting that involves a longer-range forecast horizon than regular time series forecasting.

Logically, historical observations are always available. But most models (including various Trans-
formers) infer the future by analyzing the part of past sub-series closest to the present. Longer
historical series is merely used to train model. For short-term forecasting that more concerns series
local (or call short-term) pattern, the closest sub-series carried information is enough. But not for
long-term forecasting that requires models to grasp time series’ global pattern: overall trend, long-
term seasonality, etc. Methods that only observe the recent sub-series can’t accurately distinguish
the 2 patterns and hence produce sub-optimal predictions (see Figure 1a, models observe an obvious
upward trend in the zoom-in window. But zoom out, we know that’s a yearly seasonality. And we
can see a slightly overall upward trend between the 2 years power load series). However, it’s imprac-
ticable to thoughtlessly input entire training set series as lookback window. Not only is no model yet
can tackle such a lengthy series, but learning dependence from therein is also tough. Thus, we ask:
how to enable models to inexpensively use the global information in training set during inference?

In addition, the throughput of Transformers (Zhou et al., 2022; Liu et al., 2021; Wu et al., 2021;
Zhou et al., 2021; Kitaev et al., 2020; Vaswani et al., 2017), which show the best performance in
long-term forecasting, is relatively limited, especially for fine-grained time series (e.g., recorded per
15 min, half-hour, hour). Given a common time series recorded every 15 minutes (96 time-steps per
day), with 24GB memory, they mostly fail to predict next month from 3 past months of series, even if

1Code will be released soon.

1

Under review as a conference paper at ICLR 2023

they have struggled to reduce self-attention’s computational complexity. They still can’t afford such
a length of series and thus cut down lookback window to trade off a flimsy prediction. If they are
requested to predict 3 months, how do respond? These demands are quite frequent and important in
many application fields. For fine-grained series, we argue, it has reached the bottleneck to extend the
forecast horizon through improving self-attention to be more efficient. So, in addition to modifying
self-attention, how to enhance the time series information input and output of Transformers?

We study the second question first. Prior works
process time series in a point-wise style: each
time-step in time series will be modeled in-
dividually. For Transformers, each time-step
value is mapped to a token and then calcu-
lated. This style wears out the models that en-
deavor to predict fine-grained time series over
longer-term, yet has never been questioned. In
fact, similar to images (He et al., 2022), many
time series are natural signals with temporal
information redundancy—e.g., a missing time-
step value can be interpolated from neighbor-
ing time-steps. The finer time series’ granular-
ity, the higher their redundancy and the more
accurate interpolations. Therefore, the point-
wise style is information-inefficient and waste-
ful. In order to improve the information density
of fine-grained time series, we split them into
patches and reform the point-wise to patch-wise
processing, which considerably reduces tokens and enhances information efficiency. To maintain
equivalent token information across time series with different granularity, we fix the patch size as
day. We choose the “day” as patch size because it’s moderate, close to people’s habits, and conve-
nient for modeling. Other patch sizes are also practicable, we discuss the detail in AppendixG.

Figure 1: (a) depicts two-year power load of an
area.
(b) illustrates the area’s full day load on
Jan 2, 2012, a week ago, a week later, and a year
ago: compared to a week ago or later, the power
load series of Jan 2, 2012, is closer to a year ago,
which indicates the day’s time semantics is altered
to closer to a year ago but further away from a
week ago or later.

(a) Two Year Load

(b) Full Day Load

Nevertheless, splitting time series into patches is not a silver bullet for the first question. Even
if do so, the whole training set patches series is still too lengthy. And we just want the global
information therein, for this purpose to model the whole series is not a good deal. Time is one of
the most important properties of time series and plenty of series characteristics are determined by or
affected by it. Can time be used as a container to store time series’ global information? For the whole
historical series, time is a general feature that’s very appropriate to model therein persistent temporal
patterns. Driven by this, we try to distill training set’s global information into time representations
and further substitute time series with these time representations as the main modeling entities. But
how to represent time? In Section 2, we also provide a superior method to represent time.

In this work, we challenge using merely vanilla Transformers to predict long-term series. We base
vanilla Transformers to design a brand-new forecast framework named Dateformer, it regards day
as time series atomic unit, which remarkable reduces Transformer tokens and hence improves series
information input and output, particularly for fine-grained time series. This also benefits Autore-
gressive prediction: less error accumulation and faster reasoning. Besides, to better tap training set’s
global information, we distill it, store it in the container of time representations, and take these time
representations as main modeling entities for Transformers. Dateformer achieves the state-of-the-
art performance on 7 benchmark datasets. Our main contributions are summarized as follows:

• We analyze information characteristics of time series and propose splitting time series into
patches. This considerably reduces tokens and improves series information input and out-
put thereby enabling vanilla Transformers to tackle long-term series forecasting problems.

• To better tap training set’s global information, we use time representations as containers
to distill it and take time as main modeling object. Accordingly, we design the first time-
modeling time series forecast framework exploiting vanilla Transformers—Dateformer.

• As the preliminary work, we also provide a superior time-representing method to support

the time-modeling strategy, please see section 2 for details.2

2Related works in Appendix B.

2

1yearTime400050006000Loadyearly seasonality33day40day3000400050006000upward trend?03:0006:0009:0012:0015:0018:0021:00Time300040005000600070008000LoadJan 2,2012Jan 2,2011Jan 9,2012Dec 26,2011Under review as a conference paper at ICLR 2023

2 TIME-REPRESENTING

To facilitate distilling training set’s global information, we should establish appropriate time repre-
sentations as the container to distill and store it. In this paper, we split time series into patches by
day, so we mostly study a special case of it—how to represent a date? Informer (Zhou et al., 2021)
provides a time-encoding that embeds time by stacking several time features into a vector. These
time features contain rich time semantics and hence can represent time appropriately. We follow
that and collect more date features by a set of simple algorithms3 to generate our date-embedding.

But this date-embedding is static. In practice, people pay different attention to various date features
and this attention will be dynamically adjusted as the date context change. At ordinary times, people
are more concerned about what day of week or month is today. When approaching important fes-
tivals, the attention may be shifted to other date features. For example, few people care that Jan 2,
2012, is a Monday because the day before it is New Year’s Day (see Figure 1b for an example). This
semantics shifting is similar to the polysemy in NLP field, we call it date polysemy. It reminds us: a
superior date-representation should consider the contextual information of date. However, the date
axis is extended infinitely. The contextual boundaries of date are open, which is distinct from words
that are usually located in sentences and hence have a comparatively closed context. Intuitively,
distant dates exert a weak influence on today, so it makes sense to select nearby dates as context. In-
spired by this, to encode dynamic date-representations, we introduce Date Encoder Representations
from Transformers (DERT), a convolution-style BERT (Devlin et al., 2019).

Figure 2: Date Encoder Representations from Transformers. To encoding the Jan 2, DERT selects
date-embeddings of some days before and after it as the contextual padding (the green block).

Concretely, DERT slides a fixed window Transformer encoder on the date axis to select date-
embeddings input. The input window contains date-embeddings of: some days before the target
day (pre-days padding), the target day to be represented, and some days after the target day (post-
days padding). After encoding, tokens of the pre-days and post-days paddings will be discarded,
only the target day’s token is picked out as the corresponding dynamic date-representation. We
design 2 tasks that implemented by linear layers to pre-train DERT.

Mean value prediction Predicting mean values of time series patches is a direct and effective
pre-training task, we require DERT to predict the target day’s time series mean value by the corre-
sponding date-representation. This task will lose time series patches’ intraday information, so we
design the latter task to mitigate the problem.

Auto-correlation prediction We observed that many time series display varied intraday trend and
periodicity on different kinds of days. Like more steady in holidays, or on the contrary. To leverage
it, we utilize the circular auto-correlation in stochastic process theory (Chatfield, 2003; Papoulis &
Pillai, 2002) to assess sequences’ stability. According to Wiener-Khinchin theorem (Wiener, 1930),

3see Appendix C for details

3

TransformerEncoderFFNNormNormPositional Encoding++N xEmbedded datesMulti-Head Attention+representationLinear LayerMean (correlation)predictDERTTransformer EncoderDec 30Dec 31Jan 1Jan 2Jan 3Jan 4Jan 5Jan 6Jan 7Jan 8Dec 29Dec 28Dec 27Dec 26Dec 25Jan 9Jan 10Jan 11Embed through Linear ProjectslideUnder review as a conference paper at ICLR 2023

series auto-correlation can be calculated by Fast Fourier Transforms. Thus, we Score the stability
of the target day’s time series patch X with G-length by following equations:

SX X (f ) = F(Xt)F ∗(Xt) =

(cid:90) ∞

(cid:90) ∞

Xte−i2πtf dt

Xte−i2πtf dt

RX X (τ ) =

F −1(SX X (f ))
ℓ2N orm

=

−∞

−∞
(cid:82) ∞
−∞ SX X (f )ei2πf τ df
(cid:112)X 2
2 + · · · + X 2
G

1 + X 2

(1)

Score = Score(X ) =

1
G

G−1
(cid:88)

τ =0

RX X (τ )

where F denotes Fast Fourier Transforms, ∗ and F −1 denote the conjugate and inverse Transforms
respectively. We ask DERT to predict Score of the target day’s time series patch by the day’s
date-representation.

Pre-training We use the whole training set’s time series patches to pre-train DERT, each day’s
time series patch is a training sample and the loss is calculated by:

P retraining Loss = MSE(linearLayer1(d), M ean) + MSE(linearLayer2(d), Score)
where d denotes the target day’s (patch’s) dynamic date-representation that produced by DERT,
M ean denotes true time series mean value of the day, and Score is calculated by equation 1. Our
DERT relies on supervised pre-training tasks, so it’s series-specified—different time series datasets
need to re-train respective DERT. Note that the 2 tasks are merely employed for pre-training DERT,
so they’ll be removed after pre-training and not participate in the downstream forecasting task.

3 DATEFORMER

We believe that time series fuse 2 patterns of components: global and local pattern. The global
pattern is derived from time series’ inherent characteristics and hence doesn’t fluctuate violently,
while the local pattern is caused by some accidental factors such as sudden changes in the weather,
so it’s not long-lasting. As aforementioned, the global pattern should be grasped from entire training
set series, and the local pattern can be captured from lookback window series. Based on this, we
design Dateformer, it distills training set’s global information into date-representations, and then
leverages the information to produce a global prediction. The lookback window is used to provide
local pattern information, and then contributes a local prediction. Dateformer fuses the 2 predictions
into a final prediction. To enhance Transformer’s throughput, we split time series into patches by
day, and apply Dateformer in a patch-wise processing style. Therefore, we deal with time series
forecasting problems at the day-level—predicting next H days time series from historical series.

3.1 GLOBAL PREDICTION

We distill the whole training set’s global information and store it in date-representations. Then,
during inference, we can draw a global prediction from therein to represent learned global pattern.

Date Representation = {d}

P ositional Encodings = {1, 2, 3, · · · , G}

{t1, t2, t3, · · · , tG} = Duplicate(d) + P ositional Encodings

} + {1, 2, 3, · · · , G}
= {d, d, d, · · · , d
(cid:125)

(cid:124)

(cid:123)(cid:122)
G copies

(2)

Global P rediction = FFN(t1, t2, t3, · · · , tG)

Predictive Network As equations 2, given a day’s (patch’s) dynamic date-representation d, we
duplicate it to G copies (G denotes series time-steps number every day), and add sequential posi-
tional encodings4 to these copies, thereby obtaining finer time-representations {t1, t2, t3, · · · , tG}
to represent each series time-step of the day. Then, to get the day’s global prediction, we employ a
position-wise feed-forward network to map these time-representations into time series.

4We use the canonical Positional Encoding proposed by Vaswani et al. (2017), other PEs are also practicable.

4

Under review as a conference paper at ICLR 2023

Distilling Initially, the containers of these time-representations are empty and can’t represent time
series’ global pattern. So, before formally training the entire Dateformer, we separately train the
global predictive network on training set, to distill therein global information thereby preserving the
whole training set time series’ global pattern. Each day’s time series patch in training set is a sam-
ple. Time features are sufficiently general for the whole historical series, so they won’t distill local
pattern’s ad-hoc information. For some datasets, end-to-end training Dateformer is also workable.5

3.2 LOCAL PREDICTION

The local pattern information is volatile and hence can merely be available from recent observa-
tions. To better capture the local pattern from lookback window, we eliminate the learned global
pattern from lookback window series. As described in the previous section, for a day in look-
back window, we can get the day’s global prediction from its date-representation. Then, the day’s
Series Residual r carried only local pattern information will be produced by:

Series Residual r = Series − Global P rediction

(3)

where Series denote the day’s ground-truth time series. We apply the operation to all days in look-
back window, to produce the Series Residuals {r1, r2, · · · } of the whole lookback window. Sub-
sequently, we utilize a vanilla Transformer to learn a local prediction from these Series Residuals.

Figure 3: Take Jan 2 as an example: to get the day’s local prediction, we utilize Transformer to
learn series residuals similarities Wi between Jan 2 and the lookback window (left), and use the
similarities to aggregate Series Residuals of lookback window into the local prediction (right).

Concretely, as shown in Figure 3, we feed the lookback window Series Residuals into Trans-
former encoder. For multivariate time series, pre-flattening series is required. The encoder output
will be sent into Transformer decoder as a cross information to help decoder refine input date-
representations. To learn pair-wise similarities of the forecast day and each day in lookback window,
the decoder eats date-representations of lookback window and forecast day, to exchange information
between them. Given a lookback window of P days, the similarities are calculated by:

Series Residuals = {r1, r2, · · · , rP }
Date Representations = {d1, d2, · · · , dP , dP +1}
{(cid:99)d1, (cid:99)d2, · · · , (cid:99)dP , (cid:91)dP +1} = Decoder(Date Representations,

Encoder(Series Residuals))

Similarities = {W1, W2, · · · , WP } = SoftMax(FFN((cid:99)d1, (cid:99)d2, · · · , (cid:99)dP ))

then, we can get a local prediction corresponding dP +1 by Aggregating Series Residuals:

Local P rediction = Aggregate(Series Residuals)

= W1 × r1 + W2 × r2 + · · · + WP × rP

5We discuss it in Appendix F

(4)

(5)

5

ResidualsAggregationDec 29Dec 30Dec 31Jan 1Dec 28Dec 27Dec 26Jan 2FFNDec 29Dec 30Dec 31Jan 1Dec 28Dec 27Dec 26Jan 2Date-representationsW1W2W3W4W5W6W7TransformerDecoderEmbed through Linear ProjectResiduals AggregationXXXXaggregateLocalPredictionW1W2W3W7SoftMaxTransformerEncoderSeries ResidualsDec 26Dec 26Dec 26Dec 26Dec 26Dec 27series:Dec 26Dec 26Dec 26:seriesDec 26Dec 26Dec 28:seriesDec 26Dec 26Jan 1:seriesDec 26Dec 26Dec 26Dec 26Dec 27Dec 26Dec 26Dec 28Dec 26Dec 26Dec 29Dec 26Dec 26Dec 31Dec 26Dec 26Jan 1Dec 30Flatten and Linear ProjectUnder review as a conference paper at ICLR 2023

3.3 FINAL PREDICTION

Now, we can obtain a final prediction by adding the global prediction and local prediction:

F inal P rediction = Global P rediction + Local P rediction

(6)

For multi-day forecast, we need to encode multi-day date-representations and produce their corre-
sponding global predictions and local predictions. Thus, we design Dateformer to automatically
conduct these procedures and fuse them into the final predictions, just like a scheduler.

Figure 4: Work flow of Dateformer (4-predict-2 days case), 2 pre(post)-days padding. 1⃝: sliding
DERT to encode multi-day date-representations and building their global predictions by a single
feed-forward propagation; 2⃝: making multi-day series residuals of lookback window; 3⃝, 4⃝: Au-
toregressive local predictions; 5⃝: fusing the global and local predictions into the final predictions.

Referring to Figure 4, for multi-day forecast, we input into Dateformer: i) static date-embeddings of:
pre-days padding, lookback window, forecast window, and post-days padding to encode multi-day
date-representations; ii) lookback window series to provide local information supporting the local
prediction. Dateformer slides DERT to encode dynamic date-representations of lookback and fore-
cast window. Then, the global predictions can be produced by a single feed-forward propagation.

For the local predictions, Dateformer recursively calls the Transformer to day-wise predict. Autore-
gression is applied in the procedure—the previous prediction is used for subsequent predictions:

Series Residuals = {r1, r2, · · · , rP }

Local P rediction rP +1 = Aggregate(Series Residuals)
Series Residuals = {r0, r1, r2, · · · , rP , rP +1}
Local P rediction rP +2 = Aggregate(Series Residuals) · · ·

(7)

To retard error accumulation, we adopt 3 tricks: 1) Dateformer employs Autoregression for only
local predictions, the global predictions with most numerical scale of time series are generated in a
single feed-forward propagation, which restricts the error’s scale; 2) Dateformer regards day as time
series basic unit. For fine grained time series, the day-wise Autoregression remarkable reduces errors
propagation times and accelerates prediction; 3) During local prediction Autoregression, when a day
predicted local prediction patch (such as rP +1 in equation 7) is appended to the Series Residuals
tail, Dateformer will insert earlier past a day ground-truth residual (such as r0 in equation 7) into
their head to balance the proportion between true local information and predicted residuals.

We didn’t adopt the one-step forward generative style prediction that is proposed by Zhou et al.
(2021) and followed by other Transformer-based forecast models because the style lacks scalability.
For different length forecast demands, they have to re-train models. We are committed to providing
a scalable forecast style to tackle forecast demands of various lengths. Although has defects of slow

6

Dec 27Dec 28Dec 29Dec 30Dec 31Jan 1Jan 2Jan 3 Jan 4Jan 5slidepost-dayspaddingpre-dayspaddinginput static date-embeddinglookbackwindow forecastwindowinput time-series patchesDec 29 Dec 30 Dec 31 Jan 1 Jan 2 Jan 3 Dec 26Dec 26Dec 29Dec 26Dec 26Dec 30Dec 26Dec 26Dec 31Dec 26Dec 26Jan 1Jan 2Jan 3duplicate and + PEsFFNGlobal PredictionsDate Representsseries residuals-Dec 26Dec 26Dec 29Dec 26Dec 26Dec 30Dec 26Dec 26Dec 31Dec 26Dec 26Jan 1Dec 26Dec 26Dec 29Dec 26Dec 26Dec 30Dec 26Dec 26Dec 31Dec 26Dec 26Jan 1Dec 26Dec 26Dec 29Dec 26Dec 26Dec 30Dec 26Dec 26Dec 31Dec 26Dec 26Jan 1Dec 30 Dec 31 Jan 1 Jan 2 Local PredictionTransformer Encoder Dec 30 Dec 31 Jan 1 Jan 2 Local PredictionJan 3 Transformer Decoder Transformer Decoder Transformer Encoder Jan 3Date RepresentsDec 26Dec 26Dec 30Dec 26Dec 26Dec 31Dec 26Dec 26Jan 1Dec 26Dec 26Dec 29Dec 26Dec 26Dec 30Dec 26Dec 26Dec 31Dec 26Dec 26Jan 1Jan 2Jan 2Earlier Residual1234DERTDec 26Dec 26Dec 29Dec 26Dec 26Dec 30Dec 26Dec 26Dec 31Dec 26Dec 26Jan 1Dec 26Dec 26Jan 2Dec 26Dec 26Jan 3Dec 29 +Final PredictionsGlobal PredictionsLocal Predictions5Dec 26Dec 26Jan 2Dec 26Dec 26Jan 3Jan 2Jan 3Under review as a conference paper at ICLR 2023

prediction speed and error accumulation, Autoregression can achieve the idea. So, we still adopt
Autoregression and fix its defects. Dateformer has excellent scalability. It can be trained on a short-
term forecasting setup, while robustly generalized to long-term forecasting tasks. For each dataset,
we only train Dateformer once to deal with forecast demands for any number of days.

4 EXPERIMENTS

Datasets We extensively perform experiments on 7 real-world datasets, including energy, traffic,
economics, and weather: (1) ETT (Zhou et al., 2021) dataset collects 2 years (July 2016 to July 2018)
electricity data from 2 transformers located in China, including oil temperature and load recorded
every 15 minutes or hourly; (2) ECL6 dataset collects the hourly electricity consumption data of 321
Portugal clients from 2012 to 2015; PL7 dataset contains power load series of 2 areas in China from
2009 to 2015. It’s recorded every 15 minutes and carries incomplete weather data. We eliminate
the climate information and stack the 2 series into a multivariate time-series; (4) Traffic8 contains
hourly record value of road occupancy rate in San Francisco from 2015 to 2016; (5) Weather9 is a
collection of 21 meteorological indicators series recorded every 10 minutes by a German station for
the 2020 whole year; (6) ER10 collects the daily exchange rate of 8 countries from March 2001 to
March 2022. We split all datasets into training, validation, and test set by the ratio of 6:2:2 for ETT
and PL datasets and 7:1:2 for others, just following the standard protocol.

Baselines We select 6 strong baseline models for multivariate forecast comparisons, including
4 state-of-the-art Transformer-based, 1 RNN-based, and 1 CNN-based models: FEDformer(Zhou
et al., 2022) (ICML 2022), Autoformer (Wu et al., 2021) (NeurIPS 2021), Informer (Zhou et al.,
2021) (AAAI 2021 Best Paper Award), Pyraformer (Liu et al., 2021) (ICLR 2022 Oral), LSTM
(Hochreiter & Schmidhuber, 1997), and TCN (Bai et al., 2018).

Implementation details The details of our models and Transformer-based baseline models are in
AppendixI. Our code will be released after the paper’s acceptance.

4.1 MAIN RESULTS

We show Dateformer’s performance here. Due to splitting time series by day, we evaluate models
with a wide range of forecast days: 1d, 3d, 7d, 30d, 90d (or 60d), and 180d covering short-, medium-
, and long-term forecast demands. For coarse-grained ER series, we still follow the forecast horizon
setups proposed by Wu et al. (2021). Our idea is to break the information bottleneck for models, so
we didn’t restrict models’ input. Any input length is allowed as long as the model can afford it. For
these baselines, if existing, we’ll choose the input length recommended in their paper. Otherwise, an
estimated input length will be given by us. We empirically select 7d, 9d, 12d, 26d, 46d (or 39d), and
60d as corresponding lookback days for Dateformer, and we only train it once on the 7d-predict-1d
task to test all setups for each dataset. For fairness, all these sequence-modeling baselines are trained
time-step-wise but tested day-wise. Due to the space limitation, only multivariate comparisons is
shown here, see Appendix D for univariate comparisons and Table 12 for standard deviations results.

Multivariate results Our proposed Dateformer achieves consistent state-of-the-art performance
under all forecast setups on all 7 benchmark datasets. The longer forecast horizon, the more sig-
nificant improvement. For the long-term forecasting setting (>60 days), Dateformer gives MSE
reductions: 82.5% (0.585 → 0.144, 1.672 → 0.176) in PL, 42% (1.056 → 0.690) in ETTm1,
38.6% (0.702 → 0.431) in ETTh2, 59.8% (0.316 → 0.220, 2.231 → 0.240) in ECL, 79.4%
(2.557 → 0.526) in Traffic, and 31.5% in ER. Overall, Dateformer yields a 33.6% averaged ac-
curacy improvement among all setups. Compared with other models, its errors rise mostly steadily
as forecast horizon grows. It means that Dateformer gives the most credible and robust long-range

6https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
7This dataset is provided by the 9th China Society of Electrical Engineering cup competition.
8http://pems.dot.ca.gov/
9https://www.bgc-jena.mpg.de/wetter/
10https://fred.stlouisfed.org/categories/158

7

Under review as a conference paper at ICLR 2023

Table 1: Multivariate time series forecasting results. OOM: Out Of Memory. “-” means failing to
train because validation set is too short to provide even a sample. (Number) in parentheses denotes
each dataset’s time-steps number every day. A lower MSE or MAE indicates a better prediction.
The best results are highlighted in bold and the second best results are highlighted with a underline.

Models

Dateformer

FEDformer

Autoformer

Informer

Pyraformer

LSTM

TCN

Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

0.105
0.149
0.196
0.423

0.796
0.052
0.141
0.187
0.672
0.134
0.211
0.970
0.211
0.249
2.025
0.268
0.291 OOM OOM OOM OOM OOM OOM 0.585
3.394
0.320 OOM OOM OOM OOM OOM OOM OOM OOM 3.312

0.139
0.230
0.313
0.381
0.590

0.231
0.264
0.303
0.471

0.067
0.251
0.370
0.373

0.098
0.405
0.398
0.902

0.157
0.331
0.424
0.458

0.201
0.479
0.462
0.727

0.355
0.515
0.381
0.849
0.410
1.053
0.446
1.020
0.600 OOM OOM OOM OOM OOM OOM 1.056

0.640
0.963
1.129
1.132

0.491
0.598
0.646
0.681

0.390
0.439
0.468
0.525

0.476
0.522
0.547
0.581

0.341
0.419
0.473
0.547

0.570
0.750
0.800
0.831

)
6
9
(
L
P

1
3
7
30
90
180

) 1
6
9
3
(
1
7
m
T
30
T
E
90

) 1
4
3
2
(
2
7
h
T
30
T
E
90

)
4
2
(
L
C
E

1
3
7
30
90
180

) 1
4
3
2
(
c
7
fi
f
30
a
r
T
90

) 1
4
4
3
1
(
r
7
e
h
30
t
a
e
60

W
) 96
192
336
720

1
(
R
E

0.042
0.076
0.093
0.115
0.144
0.176

0.322
0.368
0.417
0.438
0.690

0.234
0.311
0.383
0.437
0.431

0.113
0.148
0.163
0.187
0.220
0.240

0.343
0.430
0.434
0.469
0.526

0.220
0.281
0.320
0.414
0.539

0.022
0.043
0.070
0.112

0.306
0.363
0.413
0.472
0.486

0.218
0.251
0.266
0.291
0.322
0.339

0.252
0.284
0.289
0.308
0.339

0.288
0.329
0.360
0.428
0.531

0.107
0.152
0.195
0.255

0.246
0.334
0.412
0.466
0.719

0.169
0.186
0.201
0.242
0.316
-

0.547
0.581
0.613
0.652
-

0.234
0.338
0.495
0.688
-

0.041
0.062
0.087
0.165

0.327
0.381
0.426
0.483
0.618

0.288
0.302
0.316
0.351
0.404
-

0.357
0.367
0.382
0.395
-

0.304
0.375
0.472
0.580
-

0.154
0.194
0.233
0.317

0.289
0.347
0.451
0.510
0.702

0.174
0.215
0.208
0.259
0.382
-

0.554
0.625
0.705
0.686
-

0.327
0.370
0.474
0.724
-

0.040
0.061
0.096
0.389

0.364
0.395
0.451
0.511
0.631

0.293
0.329
0.320
0.364
0.439
-

0.363
0.391
0.439
0.420
-

0.377
0.402
0.461
0.604
-

0.154
0.193
0.246
0.422

1.606
1.928
6.200
4.091
2.571

0.328
0.358
0.387
0.400
0.550
-

0.678
0.721
0.768
0.959
-

0.370
0.628
1.093
2.789
-

0.248
0.430
0.756
1.073

0.577
0.553
0.698
1.115
1.359
1.462

0.709
0.890
1.028
1.240
1.010

0.723
0.873
1.111
1.052
1.436

0.425
0.474
0.634
0.834
1.048
1.159

0.622
0.878
1.114
1.089
1.128

1.478
1.417
1.469
1.804
1.766

0.520
0.558
0.705
0.729

0.111
0.197
0.216
0.698
1.415
1.672

0.734
0.833
0.881
1.033
1.081

1.126
1.900
4.410
2.919
3.356

0.365
0.434
0.362
0.442
0.501
-

0.814
1.582
0.803
1.120
-

0.267
0.468
0.523
0.878
-

0.482
0.496
0.557
0.677

0.211
0.296
0.304
0.649
0.956
1.004

0.667
0.722
0.729
0.811
0.821

0.874
1.138
1.741
1.503
1.503

0.436
0.471
0.432
0.484
0.513
-

0.480
0.743
0.444
0.615
-

0.328
0.452
0.510
0.711
-

0.541
0.558
0.586
0.652

0.505
0.713
0.818
0.806
0.845

0.498
0.832
1.747
1.869
1.511

1.058
1.505
1.897
2.970
1.855

1.074
1.583
2.429
2.014
3.754

0.997
1.111
2.024
1.717
1.217

0.412
1.140
4.877
4.674
3.330

0.268
0.308
0.281
0.288

0.368
0.412
0.475
0.430
0.786
0.459
0.460
1.304
0.552 OOM OOM 1.815
2.231

0.372
0.388
0.381
0.382

-

-

-

0.600
0.635
0.639

1.220
0.383
1.902
0.404
0.425
2.509
0.534 OOM OOM 2.413
2.557

0.337
0.358
0.356

-

-

-

0.424
0.560
0.768
1.303
-

0.361
0.474
0.642
0.773

0.231
0.388
0.442
0.823
-

0.194
0.337
0.607
0.963

0.310
0.427
0.460
0.676
-

0.340
0.440
0.600
0.772

5.868
4.232
5.727
7.141
8.168

0.452
0.536
0.825
0.866

forecast. Note that our Dateformer still contributes the best predictions on the ER series which is
coarse-grained time series without obvious periodicity.

4.2 ANALYSIS

We try to analyze the 2 forecast components’ contributions to the final prediction and explain why
Dateformer’s predictive errors rise so slow as forecast days extend. We use separate global predic-
tion or local prediction component to predict and compare their results, as shown below.

Table 2: Multivariate time series forecasting comparison of different forecast components results.
The results highlighted in bold indicate which component contributes the best prediction.

Datasets

Forecast Days

Global

MSE
Prediction MAE

Local

MSE
Prediction MAE

Final

MSE
Prediction MAE

1

0.129
0.264

0.029
0.115

0.042
0.141

PL

30

0.132
0.268

0.406
0.457

0.115
0.249

180

0.156
0.297

1.386
0.908

0.176
0.320

ECL

30

0.303
0.392

0.249
0.327

0.187
0.291

180

0.302
0.392

0.378
0.412

0.240
0.339

1

0.634
0.355

0.330
0.239

0.343
0.252

Traffic

7

0.640
0.356

0.545
0.341

0.434
0.289

90

0.656
0.362

0.725
0.414

0.526
0.339

1

0.310
0.395

0.101
0.201

0.113
0.218

8

Under review as a conference paper at ICLR 2023

It can be seen that errors of the separate global prediction hardly rise as forecast horizon grows. The
separate local prediction, however, is rapidly deteriorating. This may prove our hypothesis that time
series have 2 patterns of components: global and local pattern. As stated in section 3, the global
pattern will not fluctuate violently, while the local pattern is not long-lasting. Therefore, as forecast
horizon grows, global prediction that still keep stable errors is especially important for long-term
forecasting. Although do best in short-term forecasting, errors of local prediction increase rapidly as
forecast days extend because local pattern is not long-lasting. As time goes on, the local pattern shifts
gradually. For distant future days, current local pattern even degenerates into a noise that encumbers
predictions: comparing the 180 days prediction cases on PL, the best prediction is contributed by the
separate global prediction instead of the final prediction that disturbed by the noise. Not only local
pattern, but Dateformer also grasps global pattern, so its errors rise mostly steadily. To intuitively
understand the 2 patterns’ contributions, we use Auto-correlation that presents the same periodicity
with source series to analyze the seasonality and trend of ETT oil temperature series.

(a) Ground-Truth

(b) Global Prediction

(c) Series Residuals

(d) Remainder

Figure 5: Auto-correlation of four time series from ETT oil temperature series: (a) Ground-truth;
(b) Global Prediction; (c) Ground-truth − Global Prediction; (d) Ground-truth − Final Prediction.

As shown in Figure 5a, the ETT oil temperature series mixes complex multi-scale seasonality, and
the global prediction almost captures it perfectly in 5b. But see Figure 5c, we can find global
prediction fail to accurately estimates series mean: although a sharp descent occurs in the left end
after eliminate the global prediction, it didn’t immediately drop to near 0. Because local pattern
will effect series trends. The local pattern is not long-lasting, so we can only approximate it from
lookback window. After subtracting the final prediction, we get an Auto-correlation series of random
oscillations around 0 in 5d, which indicates source series is as unpredictable as white noise series.

4.3 ABLATION STUDIES

We conduct ablation studies about input lookback window length and proposed time-representing
method, related results and discussions are in Appendix E. Due to the space limitation, we only
list major finds here. (1) Our Dateformer can benefit from longer lookback windows, but not for
other Transformers; (2) The proposed dynamic time-representing method can effectively enhance
Dateformer’s performance; (3) But sequence-modeling Transformers, which underestimate the sig-
nificance of time and let it just play an auxiliary role, may not benefit from better time-encodings.

In addition, we also conduct ablation studies about 2 pre-training tasks that are presented in section 2.
Beyond all doubt, the pre-training task of mean prediction is naturally effective. Removing this task
to pre-train, Dateformer’s predictions remarkable deteriorates. But for some datasets, pre-training
DERT on only this task also damages Dateformer’s performance. Auto-correlation prediction can in-
duce DERT to concern time series intraday information, thereby encoding finer date-representations.

5 CONCLUSIONS

In this paper, we challenge employing merely vanilla Transformers to predict long-term series. We
analyze information characteristic of time series and propose splitting time series into patches to
enhance Transformer’s throughout. Besides, we distill the whole training set’s global information
and store it in time representations, to help the model grasp time series’ global pattern. The proposed
Dateformer yields consistent state-of-the-art performance on extensive 7 real-world datasets.

9

1yearTime-Lag0.00.20.40.60.81.0Auto-Correlation177day184day0.660.680.700.720.741yearTime-Lag0.00.20.40.60.81.0177day184day0.660.680.700.720.741yearTime-Lag0.00.20.40.60.81.04day11day0.270.290.310.330.350.370.390.410.431yearTime-Lag0.00.20.40.60.81.0Auto-Correlation4day11day0.120.090.060.030.000.030.060.090.12Under review as a conference paper at ICLR 2023

REFERENCES

Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional

and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018.

George EP Box and Gwilym M Jenkins. Some recent advances in forecasting and control. Journal

of the Royal Statistical Society. Series C (Applied Statistics), 1968.

Chris Chatfield. The analysis of time series: an introduction. Chapman and hall/CRC, 2003.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of
gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep

Bidirectional Transformers for Language Understanding. arXiv:1810.04805 [cs], 2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An
image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint
arXiv:2010.11929, 2020.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked au-
toencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 16000–16009, 2022.

Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural computation, 9(8):

1735–1780, 1997.

Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. arXiv

preprint arXiv:2001.04451, 2020.

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term
temporal patterns with deep neural networks. In The 41st International ACM SIGIR Conference
on Research & Development in Information Retrieval, pp. 95–104, 2018.

Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng
Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series
forecasting. Advances in Neural Information Processing Systems, 32, 2019.

Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar.
Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and fore-
casting. In International Conference on Learning Representations, 2021.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization.

arXiv preprint

arXiv:1711.05101, 2017.

Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis
expansion analysis for interpretable time series forecasting. arXiv preprint arXiv:1905.10437,
2019.

Athanasios Papoulis and S Unnikrishna Pillai. Probability, random variables, and stochastic pro-

cesses. Tata McGraw-Hill Education, 2002.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-
performance deep learning library. Advances in neural information processing systems, 2019.

Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, and Garrison Cottrell. A
dual-stage attention-based recurrent neural network for time series prediction. arXiv preprint
arXiv:1704.02971, 2017.

David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic fore-
casting with autoregressive recurrent networks. International Journal of Forecasting, 36(3):1181–
1191, 2020.

10

Under review as a conference paper at ICLR 2023

Rajat Sen, Hsiang-Fu Yu, and Inderjit S Dhillon. Think globally, act locally: A deep neural network
approach to high-dimensional time series forecasting. Advances in neural information processing
systems, 32, 2019.

Leslie N Smith and Nicholay Topin. Super-convergence: Very fast training of neural networks using
large learning rates. In Artificial intelligence and machine learning for multi-domain operations
applications, volume 11006, pp. 1100612. International Society for Optics and Photonics, 2019.

Sean J Taylor and Benjamin Letham. Forecasting at scale. The American Statistician, 2018.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural informa-
tion processing systems, 30, 2017.

Norbert Wiener. Generalized harmonic analysis. Acta mathematica, 55(1):117–258, 1930.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition trans-
formers with auto-correlation for long-term series forecasting. Advances in Neural Information
Processing Systems, 34, 2021.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings
of AAAI, 2021.

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency
enhanced decomposed transformer for long-term series forecasting. In Proc. 39th International
Conference on Machine Learning (ICML 2022), 2022.

11

Under review as a conference paper at ICLR 2023

A FUTURE WORKS

In this paper, as the preliminary work, we briefly introduced the dynamic date-representation and
DERT, but didn’t talk about their transferability. Actually, under certain conditions, a DERT
that is pre-trained on a time series dataset can transfer to other similar datasets to encode date-
representations. For example, PL and ETT are similar electrical series datasets recorded in different
cities of China, PL range from 2009 to 2015 while ETT range from 2016 to 2018, but we found the
DERT that is pre-trained on PL can directly transfer to forecast tasks on ETT. More interestingly,
with the transferred DERT, Dateformer demonstrates a fairly good few sample learning potential.
On the ETT series, even though only one-third of training samples are provided, Dateformer still
can converge to the same performance level as the full training samples provided (see Table 3). The
details are not completely clear yet, we will further study in future works.

Table 3: Few sample learning multivariate results on ETTm1, only MSE is reported.

Models Dateformer-ETTm1-pre-trained

Dateformer-PL-pre-trained

Samples

353

233

113

1

353

233

113

1

1
3
7
30
90

0.326
0.370
0.418
0.443
0.702

0.331
0.374
0.422
0.442
0.711

0.333
0.383
0.427
0.435
0.680

0.422
0.445
0.478
0.476
0.698

0.322
0.368
0.417
0.438
0.690

0.324
0.365
0.414
0.437
0.680

0.332
0.381
0.426
0.435
0.678

0.373
0.399
0.443
0.453
0.696

Referring to Table 3, we train 2 Dateformers on ETT series ranging full 12 months (353 samples),
8 months (233 samples), 4 months (113 samples) and 8 days (1 sample), to test all forecast horizon
setups on the same test set. The Dateformer whose DERT is pre-trained on PL series shows the same
performance level, even a slightly better because PL is a bigger dataset with a longer time span. And
fewer samples are required: 4 months of ETT series is enough for Dateformer to convergence.

B RELATED WORKS

Time series forecasting is an enduring research topic, and numerous works have been developed to
deal with the task. They can be roughly divided into two categories: statistical methods and deep
learning models. ARIMA (Box & Jenkins, 1968), Prophet (Taylor & Letham, 2018), and the filtering
methods are representative methods of the former. The deep learning models mainly include RNN-
based (Recurrent Neural Network based), CNN-based (Convolutional Neural Network based), and
Transformer-based structures.

LSTM (Hochreiter & Schmidhuber, 1997) and GRU (Chung et al., 2014) employ gating mecha-
nisms to extend series dependence learning distance and relieve the gradient vanishing or explosion
of RNN. DeepAR (Salinas et al., 2020) further combines LSTM and Autoregression for time se-
ries probabilistic distribution forecasting. LSTNet (Lai et al., 2018) adopts CNN and recurrent-skip
connections to modeling short- and long-term patterns of time series. Some RNN-based works
introduced the attention mechanism to capture long-range temporal dependence, so as to improve
predictions (Qin et al., 2017). However, RNN’s intrinsic flaws (difficulty capturing long-range de-
pendence, slow reasoning, and accumulating error) prevent them from predicting long-term series.

Temporal convolutional network (TCN) (Bai et al., 2018) is CNN-based representative work, which
models temporal dependence with the causal convolution. DeepGLO (Sen et al., 2019) also men-
tioned the concept of global and local, and employs TCN to model them. Nevertheless, the global
concept in their paper refers to relationships between related other time series, which is dis-
tinct from this paper refers to the global historical observation on the series itself.

Transformer (Vaswani et al., 2017) recently becomes popular in long-term series forecasting, owing
to its excellent long-range dependence capturing capability. But directly applying Transformer to
long-term series forecasting is computationally prohibitive due to inside self-attention’s quadratic
complexity about sequence length in both memory and time. Many studies are proposed to re-
solve the problem. LogTrans (Li et al., 2019) presented LogSparse attention that reduces the

12

Under review as a conference paper at ICLR 2023

computational complexity to O(L(logL)2), and Reformer (Kitaev et al., 2020) presented local-
sensitive hashing (LSH) attention with O(LlogL) complexity. Informer (Zhou et al., 2021) pro-
posed probSparse attention of O(LlogL) complexity, and further renovated the vanilla Transformer
architecture. Autoformer (Wu et al., 2021) and FEDformer (Zhou et al., 2022) built decomposed
blocks in Transformer and introduced low complexity enhanced attention in frequency domain or
Auto-correlation to replace self-attention. Pyraformer (Liu et al., 2021) adopted hierarchical multi-
resolution PAM attention to achieve O(L) complexity and learn multi-scale temporal dependence.
To extend forecast horizon, all these variant Transformers designed various modified self-attentions
or substitutes to reduce computational complexity, so as to tackle longer time series.

In CV community, ViT (Dosovitskiy et al., 2020) split images into patches, thereby enabling vanilla
Transformers to tackle abundant pixels. Compared to highly abstract human language, we argue
that information characteristic of time series is more similar to it of images. They both are natural
signals and exist numerical continuity between adjacent signals. Inspired by this, we follow ViT to
split time series into patches, thereby enhancing time series forecasting Transformers’ throughput,
and enabling vanilla Transformers to predict long-term series.

In addition to these structures, there are also other neural network methods such as N-BEATS (Ore-
shkin et al., 2019) that uses decomposition. Above most forecast methods can be summarized as
learning a mapping from lookback window to forecast window, and they some leverage time feature
assist prediction. However, in the end, they are confined by information because they can only rely
on the local information in lookback window. To our best knowledge, we are the first time series
forecasting work that distills the whole training set’s global information into time-representations
and takes time instead of series as the primary modeling entities—time-modeling strategy.

C DATE EMBEDDING

Observing the movement of celestial planets, changes in temperature, or the rise and fall of plants,
our forefathers distilled a set of laws, that is calendars. A wealth of wisdom is encapsulated in
the calendar. People’s activities are guided by the calendar and we believe that’s the fundamental
source of seasonality in many time series. We try to introduce the wisdom in our models as a priori
knowledge. We use the Gregorian calendar as solar calendar and the traditional Chinese calendar
as lunar calendar to deduce our date-embedding. Besides, the vacation and weekday information is
also taken into count. In our code, we provide the date-embeddings range from 2001 to 2023 for
12 countries or regions: Australia, British, Canada, China, Germany, Japan, New Zealand, Portugal,
Singapore, Switzerland, USA, and San Francisco.

Our static date-embedding stacks the following date features: abs day, year, day (month day),
year day, weekofyear,
lunar year day, dayofyear, dayof-
month, monthofyear, dayofweek, dayoflunaryear, dayoflunarmonth, monthoflunaryear, jieqiofyear,
jieqi day, dayofjieqi, holidays, workdays, residual holiday, residual workday. As an example, we
embed today by following equations:

lunar month,

lunar year,

lunar day,

abs day =

days that have passed from December 31, 2000
365.25 ∗ 5

year =

day =

year day =

weekof year =

lunar year =

lunar month =

this year − 1998.5
25

days that have passed in this month
31
days that have passed in this year
366
weeks that have passed in this year
54

this lunar year − 1998.5
25

this lunar month
12

lunar day =

days that have passed in this lunar month
30

13

Under review as a conference paper at ICLR 2023

lunar year day =

dayof year =

dayof month =

monthof year =

dayof week =

dayof lunaryear =

dayof lunarmonth =

days that have passed in this lunar year
384
days that have passed in this year − 1
total days in this year − 1
days that have passed in this month − 1
total days in this month − 1
months that have passed in this year − 1
11
days that have passed in this week − 1
6

− 0.5

− 0.5

− 0.5

− 0.5

days that have passed in this lunar year − 1
total days in this lunar year − 1
days that have passed in this lunar month − 1
total days in this lunar month − 1

− 0.5

− 0.5

monthof lunaryear =

jieqiof year =

lunar months that have passed in this lunar year − 1
11

solar terms that have passed in this year − 1
23

− 0.5

− 0.5

jieqi day =

days that have passed in this solar term
15

dayof jieqi =

days that have passed in this solar term − 1
total days in this solar term − 1

− 0.5

We standardize these date features by respective corresponding maximum. For unbounded features
(e.g., abs day and year), we use a large number to standardize them. For example, we can use 100
to standardize year, then need not worry about it in our lifetime. With DERT encoding, our model
shows excellent generalization when facing a date never seen before (See Table 2). Because too
complicated, some date features’ equations are omitted. Email authors if interested.

D UNIVARIATE RESULTS

Baselines We also select 6 strong baseline models for univariate forecast comparisons, covering
state-of-the-art deep learning models and classic statistical methods: FEDformer(Zhou et al., 2022),
Autoformer (Wu et al., 2021), Informer (Zhou et al., 2021), N-BEATS (Oreshkin et al., 2019),
DeepAR (Salinas et al., 2020), and ARIMA (Box & Jenkins, 1968).

Table 4: Univariate series forecast results. A lower MSE or MAE indicates a better prediction. The
best results are highlighted in bold and the second best results are highlighted with a underline.

Models

Dateformer

FEDformer

Autoformer

Informer

N-BEATS

DeepAR

ARIMA

Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

)
6
9
(
L
P

1
3
7
30
90
180

) 1
4
3
2
(
c
7
fi
f
30
a
r
T
90

96
192
336
720

)
1
(
R
E

0.040
0.087
0.103
0.113
0.135
0.162

0.097
0.118
0.122
0.134
0.157

0.042
0.079
0.118
0.187

0.132
0.183
0.236
0.516

0.133
0.092
0.211
0.190
0.209
0.543
0.212
0.242
0.554
0.239
0.430
1.758
0.276 OOM OOM OOM OOM OOM OOM 0.442
1.311
0.305 OOM OOM OOM OOM OOM OOM OOM OOM 1.932

0.170
0.272
0.295
0.448
0.517

0.144
0.277
0.399
0.470

0.249
0.290
0.326
0.507

0.058
0.197
0.371
0.418

0.145
0.350
0.491
0.841

0.240
0.416
0.504
0.696

0.167
0.199
0.204
0.221
0.236

0.157
0.223
0.267
0.344

0.181
0.231
0.228
0.252
-

0.069
0.108
0.149
0.209

0.288
0.342
0.329
0.349
-

0.206
0.266
0.312
0.373

0.257
0.293
0.343
0.294
-

0.097
0.124
0.144
0.297

0.361
0.390
0.425
0.389
-

0.238
0.281
0.305
0.439

0.211
0.232
0.250
0.302
-

0.341
0.453
0.695
2.540

0.304
0.327
0.342
0.393
-

0.508
0.581
0.738
1.522

0.139
0.144
0.162
0.200
-

0.235
1.697
5.616
5.154

0.229
0.236
0.260
0.306
-

0.378
1.112
2.119
1.767

0.326
0.441
0.839
0.518
0.695

0.179
0.427
0.535
1.221

14

0.272
0.506
0.512
1.063
0.863
1.115

0.399
0.457
0.683
0.503
0.617

0.342
0.584
0.659
1.012

0.996
1.163
1.325
5.594
7.801
9.824

0.461
0.796
1.240
1.627
1.739

0.086
0.213
0.191
0.267

0.801
0.878
0.926
1.292
1.739
2.039

0.499
0.703
0.915
1.086
1.130

0.190
0.282
0.320
0.364

Under review as a conference paper at ICLR 2023

Univariate results
In the univariate time series forecasting setting, Dateformer still achieves con-
sistent state-of-the-art performance under all forecast setups. For the long-term forecasting set-
ting, Dateformer gives MSE reduction: 82.3% (0.442 → 0.135, 1.932 → 0.162) in PL, 77.4%
(0.695 → 0.157) in Traffic, 23.7% in ER. Overall, Dateformer yields a 41.6% averaged accuracy
improvement among all univariate forecast setups on the given 3 representative datasets.

E ABLATION STUDIES

E.1

IMPACT OF INPUT LENGTH

In this section, we study the impact on several Transformer-based models of different input lengths.
On representative 8 forecast tasks, which cover short-, middle-, and long-term forecasting on various
time series datasets, we gradually extend their lookback window and record their predictive errors
of different input window sizes. The results are shown in the form of line charts as follows.

(a) ETTh1 7-day

(b) ETTh2 1-day

(c) ETTm1 3-day

(d) PL 7-day

(e) ECL 30-day

(f) Traffic 3-day

(g) Weather 1-day

(h) ER 720-day

Figure 6: The errors lines of several Transformers with different input lookback days of: predicting
next (a) 7-day on ETTh1,(b) 1-day on ETTh2,(c) 3-day on ETTm1,(d) 7-day on PL,(e) 30-day on
ECL,(f) 3-day on Traffic,(g) 1-day on Weather,(h) 720-day on ER time series.

As shown in Figure 6, with lookback window extends, Dateformer’s predictive errors gradually
reduce until stable, which indicates that Dateformer can leverage more information to continuously
improve prediction. Although not the best under narrow lookback windows, Dateformer finally
outperforms other all baseline Transformers as lookback window grows. But other Transformers
may not benefit from larger input windows. Their performances are unstable, and even deteriorate
as input length extends. This is against common sense: more information intake should lead to better
predictions. Compared to them, the information utilization upper limit of Dateformer is higher. We
can always expect better predictions by feeding longer lookback window series, at least not worse.

E.2

IMPACT OF TIME ENCODING

In this paper, we contribute 2 time-encodings: the static date-embedding that is generated by stack-
ing more date features straightforwardly, and the dynamic date-representation that further considers
date contextual information. So, we try to clarify which time-encoding is better, and how well other
Transformers perform when better time-encodings are provided.

15

1234567814152130Input Days0.40.60.81.01.21.41.6MSEDateformerPyraformerFEDformerAutoformerInformerReformer1234567814152130Input Days0.00.20.40.60.81.01.21.41.6MSEDateformerPyraformerFEDformerAutoformerInformerReformer1234567814152130Input Days0.500.751.001.251.501.752.00MSEDateformerPyraformerFEDformerAutoformerInformerReformer1234567814152130Input Days0.100.150.200.250.300.350.400.450.50MSEDateformerPyraformerFEDformerAutoformerInformerReformer13781415212630314560Input Days0.20.40.60.81.0MSEDateformerPyraformerFEDformerAutoformerInformerReformer1234567814152130Input Days0.40.60.81.01.21.4MSEDateformerPyraformerFEDformerAutoformerInformerReformer1234567814152130Input Days0.20.30.40.50.60.70.80.91.0MSEDateformerPyraformerFEDformerAutoformerInformerReformer1224364860728496108120336720Input Days0.20.40.60.81.01.21.4MSEDateformerPyraformerFEDformerAutoformerInformerReformerUnder review as a conference paper at ICLR 2023

E.2.1 WHICH TIME ENCODING BETTER?

The better time-encoding should be able to represent time more accurately and accommodate more
time series global information. Thus, we employ the 2 time-encoding to distill time series’ global
information through 2 global prediction components with roughly the same number of parameters,
and then check their global prediction quality. The results are shown below.

Table 5: Global prediction multivariate comparison using 2 time-encoding.

Datasets

Forecast Days

Dynamic

MSE
Representation MAE

Static
Embedding

MSE
MAE

1

0.129
0.264

0.177
0.306

PL

30

0.132
0.268

0.181
0.310

180

0.156
0.297

0.193
0.325

1

0.310
0.395

0.316
0.398

ECL

30

0.303
0.392

0.306
0.393

180

0.302
0.392

0.305
0.393

1

0.634
0.355

0.655
0.363

Traffic

7

0.640
0.356

0.659
0.362

90

0.656
0.362

0.669
0.363

As shown in Table 5, the global prediction component that employs dynamic date-representation
always contributes a better global prediction. This is enough to prove the effectiveness of our pro-
posed dynamic time-representing method. In order to more intuitively show the importance of date
contextual information, and how DERT pays attention to date context, we visualize some attention
weight distributions of DERT encoder. The figures are shown as follows.

(a) Jan 2

(b) Jan 24

(c) Apr 5

(d) Oct 2

Figure 7: DERT encoder attention weight distributions from PL dataset during 2012, the darker the
cell, the greater the attention weight. (a) is encoding Jan 2, Jan 1 is New Year’s day; (b) is encoding
Jan 24, in traditional Chinese lunar calendar, Jan 22 is New Year’s Eve and Jan 23 is the Spring
Festival; (c) is encoding Apr 5, Apr 4 is the Qingming Festival. (d) is encoding Oct 2, Sep 30 is the
Mid-Autumn Festival in Chinese lunar calendar, and Oct 1 is National Day in Gregorian calendar.

In Figure 7, when approaching important festivals, DERT pays more attention to these festivals. The
more important festival, the more concentrated attention. In China, the Spring Festival is the most
important festival of a year. So, we can see the most intensive attention on the day in Figure 7b, and
other days barely receive little attention. But referring to Figure 7c , although very intensive attention
on Apr 4, some attention is still distributed to other days. Because the Qingming Festival is not as
important as the Spring Festival. DERT will not pay excessive attention to it. This is completely in
line with Chinese habits. Besides, DERT can also tackle the changeable date semantics in different
calendars. In the traditional Chinese lunar calendar, Sep 30, 2012, is the Mid-Autumn Festival while
Oct 1 is Chinese National Day in the Gregorian calendar. The interval between the 2 festivals change

16

Dec 26Dec 27Dec 28Dec 29Dec 30Dec 31Jan 1Jan 2Jan 3Jan 4Jan 5Jan 6Jan 7Jan 8Jan 9Jan 10Jan 11Jan 12Jan 13Jan 14Jan 15Jan 16head1head2head3head4head5head6head7head8Jan 17Jan 18Jan 19Jan 20Jan 21Jan 22Jan 23Jan 24Jan 25Jan 26Jan 27Jan 28Jan 29Jan 30Jan 31Feb 1Feb 2Feb 3Feb 4Feb 5Feb 6Feb 7head1head2head3head4head5head6head7head8Mar 29Mar 30Mar 31Apr 1Apr 2Apr 3Apr 4Apr 5Apr 6Apr 7Apr 8Apr 9Apr 10Apr 11Apr 12Apr 13Apr 14Apr 15Apr 16Apr 17Apr 18Apr 19head1head2head3head4head5head6head7head8Sep 25Sep 26Sep 27Sep 28Sep 29Sep 30Oct 1Oct 2Oct 3Oct 4Oct 5Oct 6Oct 7Oct 8Oct 9Oct 10Oct 11Oct 12Oct 13Oct 14Oct 15Oct 16head1head2head3head4head5head6head7head8Under review as a conference paper at ICLR 2023

every year, and in some years they are even on the same day. In Figure 7d, they both exert influence
on Oct 2. We can not sure how their influence change when they are on the same day or on adjacent
days, so let DERT learn from data. We also call the similar problems as date polysemy. Note that
we didn’t tell DERT which day is a festival, all knowledge about festivals is learned by itself.

E.2.2 HOW WELL OTHER TRANSFORMERS PERFORM USING BETTER TIME ENCODINGS?

These Transformer-based baselines for comparison also leverage time features to assistant predic-
tions. They adopt the time embedding that proposed by Zhou et al. (2021). We are inspired by it
too, so follow it and stack more date features into our static date-embedding. In this section, we try
to clarify how well they perform when better time-encodings are provided. We modify these Trans-
formers to employ the time-encodings mentioned in the previous section, to compare the impact of
different time-encoding on them. To better understand the role of time for these sequence-modeling
Transformers, we eliminate time embedding as a comparison. The results are as follows.

Table 6: Ablation of time-encoding on ETTh1 multivariate forecast tasks with MSE metric. Without
eliminates time-encoding. Origin adopts time-encoding proposed by Zhou et al. (2021). Static and
Dynamic denote the static date-embedding and dynamic date-representation respectively. Note that
Origin and Static are essentially the same, Static just simply collects more time features.

Forecat Days

3

Models

Without

Origin

FEDformer

Autoformer

Informer

Pyraformer

Reformer

0.359

0.415

0.695

0.606

0.723

Dateformer

-

0.360

0.436

0.788

0.605

0.776

0.386

Static

0.374

0.441

0.747

0.600

0.840

0.413

Dynamic Without

Origin

90

0.367

0.422

0.872

0.591

0.699

0.369

0.853

1.029

1.395

1.064

1.179

-

1.027

0.929

1.291

1.100

1.210

0.721

Static

1.250

1.327

1.168

1.079

1.245

0.711

Dynamic

1.142

1.257

1.672

0.987

1.085

0.691

Table 7: Ablation of time-encoding on Traffic multivariate forecast tasks with MSE metric. - means
can’t train, because time-encoding is necessary for Dateformer. Best results are highlighted in bold.

Forecat Days

7

Models

Without

Origin

FEDformer

Autoformer

Informer

Pyraformer

Reformer

0.626

0.807

0.751

0.646

0.978

Dateformer

-

0.613

0.705

0.768

0.639

0.720

0.483

Static

0.601

0.653

0.787

0.644

0.703

0.451

Dynamic Without

Origin

30

Static

0.630

0.673

1.196

0.652

0.686

0.959

OOM OOM

0.683

0.510

0.721

0.487

Dynamic

0.624

0.667

1.072

OOM

0.885

0.475

0.592

0.641

0.774

0.671

0.696

0.434

0.676

1.285

0.966

OOM

1.470

-

Referring to Table 6 and 7, as a time-modeling method that mainly models time, Dateformer consis-
tently benefits from better time-encodings. However, better time-encodings may not enhance other
Transformers that mainly model series, and they only leverage time-encoding in an auxiliary style
just like positional encoding. In some forecast cases, better time-encodings can improve their pre-
dictions. But in some other cases, eliminating time-encoding results in the best prediction. Their
utilization of time is unstable. Besides, simply collecting more time features does not necessarily
work, sometimes even encumber predictions (see 3 days prediction case in Table 6).

17

Under review as a conference paper at ICLR 2023

E.3 ABLATION STUDY ABOUT PRE-TRAINING TASKS

In section 2, we design 2 pre-training tasks for DERT: mean prediction and Auto-correlation predic-
tion. Here, we check their effectiveness. We use the separate pre-training task to train DERT, and
compare their prediction results. The results are shown as follows.

Table 8: Multivariate forecast comparison of Dateformer using different pre-training tasks. Mean
adopts only pre-training task of mean prediction. Auto employs separate Auto-correlation prediction
to pre-train DERT. Both combines the 2 pre-training tasks together, and it’s adopted by us. The best
results are highlighted in bold. “-” denotes lacking test samples to report result.

Time Series Datasets

Forecast Days Metric Mean

1

3

7

30

90

180

MSE
MAE

MSE
MAE

MSE
MAE

MSE
MAE

MSE
MAE

MSE
MAE

0.044
0.146

0.096
0.217

0.115
0.243

0.143
0.284

0.165
0.311

0.189
0.333

PL

Auto

0.041
0.138

0.094
0.207

0.115
0.236

0.142
0.277

0.171
0.314

0.204
0.344

Both Mean

0.042
0.141

0.076
0.187

0.093
0.211

0.115
0.249

0.144
0.291

0.176
0.320

0.123
0.232

0.159
0.265

0.173
0.280

0.195
0.301

0.229
0.332

0.251
0.349

ECL

Auto

0.128
0.240

0.165
0.275

0.181
0.290

0.207
0.311

0.234
0.337

0.257
0.353

Both Mean

0.113
0.218

0.148
0.251

0.163
0.266

0.187
0.291

0.220
0.322

0.240
0.339

0.328
0.355

0.372
0.383

0.425
0.415

0.452
0.453

0.735
0.622

-
-

ETTm1

Auto

0.336
0.362

0.381
0.389

0.434
0.420

0.447
0.451

0.687
0.600

-
-

Both

0.326
0.356

0.370
0.382

0.418
0.411

0.443
0.448

0.702
0.600

-
-

As shown in Table 8, for pre-training DERT, the separate mean prediction or Auto-correlation pre-
diction task is effective. But combining them can obtain better date-representations. The mean
prediction will lose some intraday information of time series, so we design the Auto-correlation pre-
diction to mitigate the problem, it can induce DERT to tap some time series intraday information.

F TRAINING DETAILS

There are 3 relatively independent components in Dateformer: DERT encoder, global prediction,
and local prediction. They all can be trained separately. We found that Dateformers with different
training stages strategies have different predictive performances. For some datasets, pre-training
DERT or global prediction component is necessary. For other datasets, end-to-end training the
entire Dateformer is also feasible. Thus, we design a 3-stage training methodology to tap the full
capacity of Dateformer, and do the best on various forecast horizon demands of different datasets.

Pre-training
In section 2, we introduce 2 tasks to pre-train DERT, and the first step is to pre-train
the DERT encoder. Time series are split into patches by day to supervise the learning of DERT.
Each task contributes half of the pre-training loss. The pre-training stage aims to provide superior
time representations as a container to distill and store global information from training set.

Warm-up The second step is using the separately global prediction component to distill time
series global information from training set, we call it warm-up phase. In warm-up, the pre-trained
DERT encoder is loaded, then we train the separately global prediction component on training set.
We insert this stage to force the global prediction component to remember the global characteristics
of time series, so as to help Dateformer produce more robust long-range predictions. Furthermore,
it also serves as the adapter when transferring a DERT from other datasets. This phase is optional,
and we observed that a better short-term forecast is usually provided by the Dateformer skipping
warm-up. More credible long-term predictions, however, always draw from the preheated one.

18

Under review as a conference paper at ICLR 2023

Formal training Dateformer loads pre-trained DERT or global prediction component then start
training. We would use a small learning rate to fine-tune the pre-trained parameters. With enough
memory, Dateformer can extrapolate to any number of days in the future, once trained.

Table 9: Multivariate time series forecasting comparisons of different training strategies, where
Dateformer11 goes through all 3 stages. Dateformer10 skips the warm-up, and Dateformer01 skips
the pre-training stage. Dateformer00 is end-to-end trained. The ∗ marked group of result is selected
to compare with baseline models in the main text. The best results are highlighted in bold.

Models Dateformer11 Dateformer10 Dateformer01 Dateformer00

)
6
9
(
L
P

Metric MSE MAE MSE MAE MSE MAE MSE MAE
1 0.042∗ 0.141∗ 0.032 0.119 0.044 0.142 0.027 0.112
3 0.076∗ 0.187∗ 0.107 0.220 0.078 0.192 0.077 0.176
7 0.093∗ 0.211∗ 0.147 0.271 0.097 0.219 0.122 0.227
30 0.115∗ 0.249∗ 0.275 0.375 0.133 0.269 0.324 0.384
90 0.144∗ 0.291∗ 0.600 0.587 0.166 0.311 0.798 0.679
180 0.176∗ 0.320∗ 0.911 0.733 0.218 0.358 1.238 0.843
0.344 0.368 0.322∗ 0.355∗ 0.345 0.363 0.325 0.356
) 1
6
0.383 0.396 0.368∗ 0.381∗ 0.389 0.390 0.369 0.382
9
3
(
1
0.441 0.431 0.417∗ 0.410∗ 0.439 0.420 0.417 0.411
7
m
30 0.494 0.487 0.438∗ 0.446∗ 0.477 0.472 0.440 0.447
T
T
90 0.789 0.646 0.690∗ 0.600∗ 0.723 0.614 0.693 0.598
E
0.234 0.303 0.226 0.292 0.234∗ 0.306∗ 0.228 0.295
) 1
4
0.315 0.362 0.296 0.340 0.311∗ 0.363∗ 0.299 0.345
3
2
(
2
0.406 0.422 0.369 0.389 0.383∗ 0.413∗ 0.372 0.394
7
h
T
30 0.458 0.471 0.407 0.438 0.437∗ 0.472∗ 0.404 0.436
T
E
90 0.537 0.521 0.590 0.553 0.431∗ 0.486∗ 0.582 0.542
1 0.113∗ 0.218∗ 0.101 0.198 0.121 0.231 0.100 0.198
3 0.148∗ 0.251∗ 0.146 0.237 0.158 0.265 0.156 0.241
7 0.163∗ 0.266∗ 0.166 0.257 0.175 0.284 0.175 0.261
30 0.187∗ 0.291∗ 0.219 0.306 0.202 0.311 0.226 0.309
90 0.220∗ 0.322∗ 0.334 0.391 0.239 0.346 0.354 0.399
180 0.240∗ 0.339∗ 0.338 0.392 0.255 0.362 0.353 0.398
0.341 0.248 0.340 0.240 0.343∗ 0.252∗ 0.339 0.239
) 1
4
0.459 0.293 0.557 0.321 0.430∗ 0.284∗ 0.484 0.293
3
2
(
0.468 0.301 0.578 0.337 0.434∗ 0.289∗ 0.494 0.306
c
7
fi
f
30 0.503 0.319 0.621 0.362 0.469∗ 0.308∗ 0.543 0.332
a
r
T
90 0.559 0.344 0.685 0.389 0.526∗ 0.339∗ 0.609 0.358
0.224 0.294 0.223 0.295 0.220∗ 0.288∗ 0.223 0.286
) 1
4
4
0.289 0.338 0.285 0.338 0.281∗ 0.329∗ 0.285 0.329
3
1
(
0.332 0.370 0.326 0.369 0.320∗ 0.360∗ 0.329 0.364
r
7
e
h
30 0.427 0.435 0.419 0.438 0.414∗ 0.428∗ 0.413 0.422
t
a
e
60 0.545 0.540 0.537 0.536 0.539∗ 0.531∗ 0.524 0.524
96 0.045 0.160 0.022∗ 0.107∗ 0.036 0.135 0.022 0.107
192 0.076 0.206 0.043∗ 0.152∗ 0.065 0.181 0.043 0.152
336 0.137 0.274 0.070∗ 0.195∗ 0.093 0.221 0.070 0.195
720 0.336 0.426 0.112∗ 0.255∗ 0.138 0.282 0.113 0.256

)
4
2
(
L
C
E

)
1
(
R
E

W

We use different training stages to train Dateformer, and results are shown in Table 9. It can be
seen that different training stage setups can lead to different performances of Dateformers. There
is no versatile training strategy does best on all forecast horizons of all datasets. Generally, the
pre-training and warm-up can induce Dateformers to be more concerned with the global pattern of
time series, and produce robust long-term predictions. For some datasets, before distilling global
information from training set, pre-training a DERT encoder can enhance the distilling effect. But
for other datasets, the 2 stages can be combined into one. Actually, the global prediction is also a
fairly good pre-training task for DERT. The end-to-end trained Dateformer always contributes the
best short-term predictions. Because we train Dateformer in the short-term forecasting task of 7d-
predict-1d, the direct end-to-end training makes Dateformer more care local pattern of time series.

19

Under review as a conference paper at ICLR 2023

G PATCH SIZE CHOICE

In the main text, we choose the “day” as patch size, because it’s moderate, close to people’s habits,
and convenient for modeling. Here, we try some other patch sizes. We select 1
3 day, half-day, and
3-day as comparative patch sizes. The results are shown below.

Table 10: Multivariate forecast comparison of Dateformer using different patch sizes. “-” denotes
that the patch size is too coarse to align with the forecast horizon.

Time Series Datasets

Forecast Days Metric

1

3

7

30

90

MSE
MAE

MSE
MAE

MSE
MAE

MSE
MAE

MSE
MAE

ETTh2

Traffic

1
3 day
0.224
0.315

0.325
0.382

0.428
0.452

0.513
0.541

0.540
0.577

half-day

day

3-day

0.226
0.315

0.318
0.374

0.407
0.429

0.458
0.483

0.485
0.505

0.234
0.306

0.311
0.363

0.383
0.413

0.437
0.472

0.431
0.486

-
-

0.366
0.422

-
-

0.690
0.722

0.676
0.715

1
3 day
0.421
0.285

0.558
0.379

0.564
0.384

0.627
0.455

0.655
0.447

half-day

day

3-day

0.417
0.283

0.526
0.326

0.524
0.328

0.548
0.340

0.582
0.360

0.343
0.252

0.430
0.284

0.434
0.289

0.469
0.308

0.526
0.339

-
-

0.472
0.311

-
-

0.527
0.330

0.564
0.361

As shown in Table 10, for some time series, finer patch sizes can lead to better short-term predictions,
because the closer time series carries more accurate local pattern information. Compared to a day
ago, the local pattern of the present is more similar to it of half a day ago. But their mid-long-term
predictions deteriorate, because finer patch sizes will result in more local predictions recursions for
the same size forecast horizons, which means more error accumulation. In addition, for these time
series whose daily periodicity is dominant (like Traffic), the “day” is the best patch size. These
time series are usually closely related to human activities and hence more concerned with daily
patterns. The coarse patch size beyond “day” is not the most important activity period of people,
so it’s difficult to construct sufficiently accurate time-representations. If there is a holiday in the
3-day patch, how to embed which day it is in the time-embedding by the method mentioned at the
beginning of Section 2? That leads over-coarse patch sizes inapplicable. For most time series related
to human activity, we recommend the “day” as the patch size, it’s moderate, close to people’s habits.

H HYPER PARAMETER SENSITIVITY

We check the robustness with respect to the hyper-parameters: pre-days and post-days paddings. We
select 6 groups of paddings: (1, 1), (3, 3), (7, 7), (7, 14), (14, 14) and (30, 30). To be more intuitive,
we test the global prediction component with above several paddings on PL dataset.

Table 11: Dateformer’s multivariate global prediction results on PL dataset using several paddings.

Paddings

1, 1

3, 3

7, 7

7, 14

14, 14

30, 30

Metric

MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

1
3
7
30
90
180

0.155
0.155
0.156
0.161
0.179
0.190

0.290
0.290
0.291
0.296
0.315
0.326

0.151
0.152
0.153
0.158
0.176
0.187

0.285
0.286
0.287
0.292
0.311
0.324

0.127
0.128
0.128
0.132
0.145
0.155

0.264
0.264
0.265
0.269
0.284
0.295

0.134
0.134
0.134
0.137
0.151
0.165

0.270
0.271
0.271
0.274
0.291
0.305

0.139
0.139
0.139
0.142
0.156
0.173

0.278
0.278
0.278
0.281
0.297
0.314

0.156
0.157
0.157
0.161
0.177
0.197

0.291
0.292
0.293
0.297
0.314
0.335

As shown in Table 11, too large or too small padding sizes will make the prediction performance
worse. The moderate padding size (7, 7) leads to the best global prediction. This is also consistent
with how people generally think—we don’t simply consider the here and now, but also don’t think

20

Under review as a conference paper at ICLR 2023

too far ahead. In the main text, to balance predictive performances between various time series
datasets, we use the padding size of (7, 14) instead of the best (7, 7).

I

IMPLEMENTATION DETAILS

Our proposed models are trained with L2 loss, and using AdamW (Loshchilov & Hutter, 2017)
optimizer with weight decay of 7e−4. We adjust learning rate by OneCycleLR (Smith & Topin,
2019) which use 3 phase scheduling with the percentage of the cycle spent increasing the learning
rate is 0.4, and the max learning rate is 1e−3. Batch size for pre-training is 64, and 32 for others. The
total epochs are 100, but the models normally super converges very quickly. All experiments are
repeated 3 times, implemented by PyTorch(Paszke et al., 2019), and trained on a NVIDIA RTX3090
24GB GPUs. Numbers of the pre-days and post-days padding are 7 and 14 respectively. There is 1
layer in DERT encoder, and the inside Transformer contains 4 layers both in encoder and decoder.

The implementations of Autoformer (Wu et al., 2021), Informer (Zhou et al., 2021), and Reformer
(Kitaev et al., 2020) are from the Autoformer’s repository 11. And the implementations of FED-
former12 (Zhou et al., 2022) and Pyraformer13 (Liu et al., 2021) are from their respective repository.
We adopt the hyper-parameters setting that recommended in their repositories but unify the token’s
dimension dmodel as 512. We fix the input series length as 96 time-steps for FEDformer, Auto-
former, Informer, and Reformer. This is recommended by them or empirical results, and we found
extending their input length will result in unstable performances of these baselines (see section E.1).
For Pyraformer, facing longer forecast horizons, we extend its input length as their paper recom-
mended. For more detailed hyper-parameters setting please refer to their code repositories. For
other baseline models, we also use grid-search method to select their hyper-parameters.

J COMPLEXITY ANALYSIS

We also provide the complexity analysis of the global prediction and local prediction components.
For an input lookback window with length Li and output forecast window with length Lo, they’re
all divisible by G in the setting of splitting time series into patches by day, where G denotes time-
steps number every day of the time series datasets. The complexity of global prediction component
is O(Li + Lo). For local prediction component, it recurses Lo
G times and the time complexity of
each time shall not exceed O(( Li+Lo
G )2). So, the time complexity of local prediction component is
O( Lo
G )2
happens in the last day’s prediction. In practice, G2 will be a big number for fine-grained time
series and the Li + Lo takes up the most memory, it becomes the main factor that restricts us to
predict further away. Under 24GB memory, Dateformer’s maximum forecast horizon exceeds all
baseline models. And its inference speed is just slightly slower than Transformers that adopts one-
step forward generative style inference (Zhou et al., 2021) because the local prediction component
requires a few recursions.

G )2) at most. Its maximum memory usage is O(Li + Lo + ( Li+Lo

G )2), where ( Li+Lo

G ( Li+Lo

11https://github.com/thuml/Autoformer
12https://github.com/MAZiqing/FEDformer
13https://github.com/alipay/Pyraformer

21

Under review as a conference paper at ICLR 2023

K MAIN RESULTS WITH STANDARD DEVIATIONS

We repeat all experiments 3 times, and the results with standard deviations are shown in Table 12.

Table 12: Quantitative results with fluctuations of different forecast days for multivariate forecast.

Models

Metric

)
6
9
(
L
P

1
3
7
30
90
180

) 1
3
7
30
90

6
9
(
1
m
T
T
E

) 1
3
7
30
90

4
2
(
2
h
T
T
E

)
4
2
(
L
C
E

1
3
7
30
90
180

) 1
4
3
2
(
c
7
fi
f
30
a
r
T
90

) 1
4
4
3
1
(
r
7
e
h
30
t
a
e
60

W
) 96
192
336
720

1
(
R
E

Dateformer

FEDformer

Autoformer

Informer

Pyraformer

MSE

MAE

MSE

MAE

MSE

MAE

MSE

MAE

MSE

MAE

0.042±0.003
0.076±0.001
0.093±0.001
0.115±0.004
0.144±0.003
0.176±0.004

0.322±0.008
0.368±0.006
0.417±0.006
0.438±0.002
0.690±0.019

0.234±0.001
0.311±0.005
0.383±0.010
0.437±0.015
0.431±0.033

0.113±0.002
0.148±0.002
0.163±0.002
0.187±0.001
0.220±0.003
0.240±0.002

0.343±0.004
0.430±0.007
0.434±0.006
0.469±0.004
0.526±0.002

0.220±0.001
0.281±0.002
0.320±0.005
0.414±0.005
0.539±0.010

0.022±0.001
0.043±0.001
0.070±0.001
0.112±0.001

0.141±0.006
0.187±0.001
0.211±0.002
0.249±0.005
0.291±0.004
0.320±0.004

0.355±0.004
0.381±0.004
0.410±0.003
0.446±0.001
0.600±0.007

0.306±0.002
0.363±0.002
0.413±0.005
0.472±0.002
0.486±0.008

0.218±0.003
0.251±0.003
0.266±0.003
0.291±0.002
0.322±0.003
0.339±0.003

0.252±0.001
0.284±0.003
0.289±0.002
0.308±0.002
0.339±0.001

0.288±0.002
0.329±0.002
0.360±0.001
0.428±0.003
0.531±0.003

0.107±0.001
0.152±0.001
0.195±0.001
0.255±0.001

0.105±0.001
0.149±0.004
0.196±0.009
0.423±0.020
OOM
OOM

0.341±0.005
0.419±0.008
0.473±0.003
0.547±0.029
OOM

0.246±0.012
0.334±0.011
0.412±0.004
0.466±0.010
0.719±0.035

0.169±0.001
0.186±0.001
0.201±0.002
0.242±0.002
0.316±0.008
-

0.547±0.004
0.581±0.003
0.613±0.001
0.652±0.001
-

0.234±0.001
0.338±0.001
0.495±0.112
0.688±0.021
-

0.041±0.002
0.062±0.001
0.087±0.002
0.165±0.006

0.231±0.003
0.264±0.005
0.303±0.007
0.471±0.010
OOM
OOM

0.390±0.004
0.439±0.001
0.468±0.004
0.525±0.018
OOM

0.327±0.010
0.381±0.003
0.426±0.002
0.483±0.003
0.618±0.017

0.288±0.001
0.302±0.001
0.316±0.002
0.351±0.003
0.404±0.006
-

0.357±0.004
0.367±0.002
0.382±0.001
0.395±0.003
-

0.304±0.002
0.375±0.004
0.472±0.068
0.580±0.011
-

0.154±0.004
0.194±0.002
0.233±0.005
0.317±0.005

0.098±0.013
0.405±0.022
0.398±0.096
0.902±0.371
OOM
OOM

0.491±0.046
0.598±0.029
0.646±0.075
0.681±0.031
OOM

0.289±0.010
0.347±0.008
0.451±0.019
0.510±0.004
0.702±0.056

0.174±0.006
0.215±0.024
0.208±0.007
0.259±0.005
0.382±0.066
-

0.554±0.006
0.625±0.041
0.705±0.009
0.686±0.020
-

0.327±0.023
0.370±0.009
0.474±0.044
0.724±0.008
-

0.040±0.001
0.061±0.002
0.096±0.007
0.389±0.313

0.201±0.017
0.479±0.018
0.462±0.073
0.727±0.182
OOM
OOM

0.476±0.014
0.522±0.014
0.547±0.026
0.581±0.009
OOM

0.364±0.007
0.395±0.004
0.451±0.014
0.511±0.004
0.631±0.032

0.293±0.007
0.329±0.020
0.320±0.006
0.364±0.005
0.439±0.033
-

0.363±0.008
0.391±0.025
0.439±0.008
0.420±0.011
-

0.377±0.017
0.402±0.006
0.461±0.035
0.604±0.005
-

0.154±0.003
0.193±0.004
0.246±0.008
0.422±0.131

0.067±0.002
0.251±0.045
0.370±0.014
0.373±0.013
OOM
OOM

0.640±0.058
0.963±0.082
1.129±0.051
1.132±0.005
OOM

1.606±0.198
1.928±0.283
6.200±0.470
4.091±0.180
2.571±0.023

0.328±0.009
0.358±0.006
0.387±0.009
0.400±0.008
0.550±0.019
-

0.678±0.018
0.721±0.023
0.768±0.021
0.959±0.030
-

0.370±0.029
0.628±0.008
1.093±0.057
2.789±0.326
-

0.248±0.005
0.430±0.063
0.756±0.056
1.073±0.021

0.157±0.003
0.331±0.015
0.424±0.005
0.458±0.013
OOM
OOM

0.570±0.024
0.750±0.042
0.800±0.040
0.831±0.004
OOM

0.997±0.060
1.111±0.087
2.024±0.065
1.717±0.032
1.217±0.011

0.412±0.006
0.430±0.006
0.459±0.006
0.460±0.004
0.552±0.011
-

0.383±0.017
0.404±0.013
0.425±0.011
0.534±0.017
-

0.424±0.022
0.560±0.006
0.768±0.024
1.303±0.061
-

0.361±0.003
0.474±0.030
0.642±0.028
0.773±0.011

0.052±0.002
0.134±0.003
0.211±0.005
0.268±0.003
0.585±0.001
OOM

0.515±0.036
0.849±0.061
1.053±0.010
1.020±0.013
1.056±0.018

0.412±0.030
1.140±0.033
4.877±0.752
4.674±0.293
3.330±0.036

0.268±0.006
0.308±0.007
0.281±0.008
0.288±0.005
OOM
-

0.600±0.002
0.635±0.005
0.639±0.005
OOM
-

0.231±0.011
0.388±0.011
0.442±0.007
0.823±0.007
-

0.194±0.006
0.337±0.013
0.607±0.110
0.963±0.030

0.139±0.002
0.230±0.002
0.313±0.001
0.381±0.003
0.590±0.004
OOM

0.505±0.020
0.713±0.027
0.818±0.007
0.806±0.009
0.845±0.009

0.498±0.018
0.832±0.014
1.747±0.158
1.869±0.055
1.511±0.014

0.372±0.006
0.388±0.004
0.381±0.008
0.382±0.004
OOM
-

0.337±0.001
0.358±0.005
0.356±0.004
OOM
-

0.310±0.011
0.427±0.010
0.460±0.002
0.676±0.005
-

0.340±0.005
0.440±0.008
0.600±0.060
0.772±0.011

L SHOWCASES

In order to more intuitively show Dateformer’s prediction results, we visualized the time series
ground-truth and predictions of several forecast tasks. The charts are shown as follows.

(a) Dateformer

(b) FEDformer

(c) Autoformer

(d) Informer

Figure 8: 3 days prediction cases from ETTm1 oil temperature series

It can be seen that Dateformer’s predictions are closest to the ground-truth. Compared to other
models, Dateformer accurately grasps time series’ global pattern: e.g., overall trend and long-range
seasonality, that is what other models are not good at. Because they can only analyze lookback
window series to predict. But the global pattern should be captured from whole training set series.

22

0501001502002503003504002.22.01.81.61.4GroundTruthPrediction0501001502002503003504002.22.01.81.61.4GroundTruthPrediction0501001502002503003504002.22.01.81.61.4GroundTruthPrediction0501001502002503003504002.22.01.81.61.4GroundTruthPredictionUnder review as a conference paper at ICLR 2023

(a) Dateformer

(b) FEDformer

(c) Autoformer

(d) Informer

Figure 9: 3 days prediction cases from ETTh2 oil temperature series

(a) Dateformer

(b) FEDformer

(c) Autoformer

(d) Informer

Figure 10: 7 days prediction cases from Traffic series

(a) Dateformer

(b) FEDformer

(c) Autoformer

(d) Informer

Figure 11: 7 days prediction cases from Power Load series

(a) 30 days

(b) 90 days

(c) 90 days

(d) 180 days

Figure 12: Mid-long-term prediction cases of Dateformer from Power Load series

23

0204060801.501.251.000.750.500.250.000.25GroundTruthPrediction0204060801.501.251.000.750.500.250.000.25GroundTruthPrediction0204060801.501.251.000.750.500.250.000.25GroundTruthPrediction0204060801.501.251.000.750.500.250.000.25GroundTruthPrediction025507510012515017520010123GroundTruthPrediction025507510012515017520010123GroundTruthPrediction025507510012515017520010123GroundTruthPrediction025507510012515017520010123GroundTruthPrediction01002003004005006007008002.52.01.51.00.5GroundTruthPrediction01002003004005006007008002.52.01.51.00.5GroundTruthPrediction01002003004005006007008003.02.52.01.51.00.5GroundTruthPrediction01002003004005006007008002.502.252.001.751.501.251.000.750.50GroundTruthPrediction0500100015002000250030002.52.01.51.00.50.00.51.01.5GroundTruthPrediction0200040006000800021012GroundTruthPrediction0200040006000800010123GroundTruthPrediction02500500075001000012500150001750021012GroundTruthPredictionUnder review as a conference paper at ICLR 2023

M LOCAL PREDICTIONS WITH GENERATIVE STYLE

In Section 3.2, we introduce Dateformer’s local prediction component that implemented by the
vanilla Transformer. But note that the vanilla Transformer is just to produce similarities to aggregate
If we employ a local prediction
series residuals, it does not directly generate local prediction.
component with generative style, how does Dateformer perform? As a basic comparison, we modify
Dateformer’s local prediction component to be generative style.

• Dateformer-GD: removing Equation 4’s FFN and SoftMax that calculate similarities and
attaching a FFN to (cid:91)dP +1 to directly generate the local prediction corresponding dP +1.
• Dateformer-GR: inputting Date Representations to the Transformer’s encoder and
inputting Series Residuals to the Transformer’s decoder. For the decoder’s output
{ (cid:98)r1, (cid:98)r2, · · · , (cid:99)rP }, we use:

(cid:98)r = AveragePooling( (cid:98)r1, (cid:98)r2, · · · , (cid:99)rP )

Local P rediction = FFN((cid:98)r)

(8)

to directly generate the local prediction corresponding dP +1.

Table 13: Multivariate forecast comparison of several Dateformer variants. “-” denotes lacking test
samples to report result.

Time Series Datasets

Forecast Days Metric

1

3

7

30

90

180

MSE
MAE

MSE
MAE

MSE
MAE

MSE
MAE

MSE
MAE

MSE
MAE

PL

GD

0.047
0.153

0.093
0.211

0.127
0.252

0.218
0.350

0.297
0.433

0.464
0.540

Ours

0.042
0.141

0.076
0.187

0.093
0.211

0.115
0.249

0.144
0.291

0.176
0.320

GR

0.047
0.149

0.087
0.206

0.109
0.237

0.144
0.282

0.170
0.314

0.177
0.322

Ours

0.343
0.252

0.430
0.284

0.434
0.289

0.469
0.308

0.526
0.339

-
-

Traffic

GD

0.639
0.384

0.653
0.393

0.660
0.396

0.675
0.399

0.680
0.404

-
-

GR

0.642
0.387

0.653
0.393

0.657
0.394

0.675
0.400

0.685
0.409

-
-

As shown in Table 13, the local prediction component with aggregating style always outperforms it
with generative style. Due to splitting time series into patches, training samples considerably reduce.
That makes the local prediction component with generative style easy to overfit. To mitigate the
problem, we introduce an inductive bias: the local patterns of adjacent time series are similar and
hence design the local prediction component with aggregated style to aggregate similar local pattern
information.

24

Under review as a conference paper at ICLR 2023

N OTHER MODEL’S REMAINDERS

(a) Informer

(b) Reformer

(c) Autoformer

(d) FEDformer

Figure 13: Auto-correlation of 4 time series remainders from ETT oil temperature series.

We show 4 auto-correlation series in Figure 5, to provide a intuitive view to the characteristic of
global and local pattern. That’s not for comparison with other models, just to intuitively explain
each component’s function.

But as additional interests, we also provide auto-correlation series of other Transformers’ remainders
here. As shown in Figure 13a and 13b, the auto-correlation series of the remainders produced by
Informer and Reformer did not drop to 0 immediately at the left end, which indicates they fail to ac-
curately predict time series’ mean. And the remainders of Informer, Reformer, and Autoformer still
exhibit obvious seasonality that they didn’t capture. Referring to Figure 13d, though not obvious,
there is also weak seasonality in FEDformer’s remainder.

25

1yearTime-Lag0.00.20.40.60.81.0Auto-Correlation4day11day0.400.450.500.550.601yearTime-Lag0.00.20.40.60.81.0Auto-Correlation4day11day0.250.300.350.400.450.500.551yearTime-Lag0.00.20.40.60.81.0Auto-Correlation4day11day0.120.090.060.030.000.030.060.090.121yearTime-Lag0.00.20.40.60.81.0Auto-Correlation4day11day0.120.090.060.030.000.030.060.090.12