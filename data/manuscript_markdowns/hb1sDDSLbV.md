Published as a conference paper at ICLR 2021

LEARNING EXPLANATIONS THAT ARE HARD TO VARY

Giambattista Parascandolo1, 2, * Alexander Neitz1, *
Antonio Orvieto2
1MPI for Intelligent Systems, Tübingen,
˚equal contribution

Luigi Gresele1, 3 Bernhard Schölkopf1, 2
2ETH, Zürich,

3MPI for Biological Cybernetics, Tübingen

ABSTRACT

In this paper, we investigate the principle that good explanations are hard to vary
in the context of deep learning. We show that averaging gradients across examples
– akin to a logical OR (_) of patterns – can favor memorization and ‘patchwork’
solutions that sew together different strategies, instead of identifying invariances.
To inspect this, we ﬁrst formalize a notion of consistency for minima of the loss
surface, which measures to what extent a minimum appears only when examples
are pooled. We then propose and experimentally validate a simple alternative
algorithm based on a logical AND (^), that focuses on invariances and prevents
memorization in a set of real-world tasks. Finally, using a synthetic dataset with a
clear distinction between invariant and spurious mechanisms, we dissect learning
signals and compare this approach to well-established regularizers.

1

INTRODUCTION

Consider the top of Figure 1, which shows a view from
above of the loss surface obtained as we vary a two di-
mensional parameter vector θ “ pθ1, θ2q, for a ﬁctional
dataset containing two observations xA and xB. Note
the two global minima on the top-right and bottom-left.
Depending on the initial values of θ — marked as white
circles — gradient descent converges to one of the two
minima. Judging solely by the value of the loss function,
which is zero in both cases, the two minima look equally
good.

However, looking at the loss surfaces for xA and xB
separately, as shown below, a crucial difference between
those two minima appears: Starting from the same ini-
tial parameter conﬁgurations and following the gradient
of the loss, ∇θLpθ, xiq, the probability of ﬁnding the
same minimum on the top-right in either case is zero. In
contrast, the minimum in the lower-left corner has a sig-
niﬁcant overlap across the two loss surfaces, so gradient
descent can converge to it even if training on xA (or xB)
only. Note that after averaging there is no way to tell
what the two loss surfaces looked like: Are we destroying information that is potentially important?

Figure 1: Loss landscapes of a two-parameter
model. Averaging gradients forgoes informa-
tion that can identify patterns shared across dif-
ferent environments.

In this paper, we argue that the answer is yes. In particular, we hypothesize that if the goal is to
ﬁnd invariant mechanisms in the data, these can be identiﬁed by ﬁnding explanations (e.g. model
parameters) that are hard to vary across examples. A notion of invariance implies something that
stays the same, as something else changes. We assume that data comes from different environments:
An invariant mechanism is shared across all, generalizes out of distribution (o.o.d.), but might be hard
to model; each environment also has spurious explanations that are easy to spot (‘shortcuts’), but do
not generalize o.o.d. From the point of view of causal modeling, such invariant mechanisms can be
interpreted as conditional distributions of the targets given causal features of the inputs; invariance
of such conditionals is expected if they represent causal mechanisms, that is — stable properties of
the physical world (see e.g. Hoover (1990)). Generalizing o.o.d. means therefore that the predictor
should perform equally well on data coming from different settings, as long as they share the causal
mechanisms.

1

✓1<latexit sha1_base64="SrHi+Al9vauGKXsHgyllQXQrrvk=">AAAB73icbVDLSgNBEOyNrxhfUY9eBoPgKeyKoMegF48RzAOSJcxOepMhsw9neoUQ8hNePCji1d/x5t84SfagiQUNRVU33V1BqqQh1/12CmvrG5tbxe3Szu7e/kH58KhpkkwLbIhEJbodcINKxtggSQrbqUYeBQpbweh25reeUBuZxA80TtGP+CCWoRScrNTu0hCJ97xeueJW3TnYKvFyUoEc9V75q9tPRBZhTEJxYzqem5I/4ZqkUDgtdTODKRcjPsCOpTGP0PiT+b1TdmaVPgsTbSsmNld/T0x4ZMw4CmxnxGlolr2Z+J/XySi89icyTjPCWCwWhZlilLDZ86wvNQpSY0u40NLeysSQay7IRlSyIXjLL6+S5kXVc6ve/WWldpPHUYQTOIVz8OAKanAHdWiAAAXP8ApvzqPz4rw7H4vWgpPPHMMfOJ8/zWePzA==</latexit><latexit sha1_base64="SrHi+Al9vauGKXsHgyllQXQrrvk=">AAAB73icbVDLSgNBEOyNrxhfUY9eBoPgKeyKoMegF48RzAOSJcxOepMhsw9neoUQ8hNePCji1d/x5t84SfagiQUNRVU33V1BqqQh1/12CmvrG5tbxe3Szu7e/kH58KhpkkwLbIhEJbodcINKxtggSQrbqUYeBQpbweh25reeUBuZxA80TtGP+CCWoRScrNTu0hCJ97xeueJW3TnYKvFyUoEc9V75q9tPRBZhTEJxYzqem5I/4ZqkUDgtdTODKRcjPsCOpTGP0PiT+b1TdmaVPgsTbSsmNld/T0x4ZMw4CmxnxGlolr2Z+J/XySi89icyTjPCWCwWhZlilLDZ86wvNQpSY0u40NLeysSQay7IRlSyIXjLL6+S5kXVc6ve/WWldpPHUYQTOIVz8OAKanAHdWiAAAXP8ApvzqPz4rw7H4vWgpPPHMMfOJ8/zWePzA==</latexit><latexit sha1_base64="SrHi+Al9vauGKXsHgyllQXQrrvk=">AAAB73icbVDLSgNBEOyNrxhfUY9eBoPgKeyKoMegF48RzAOSJcxOepMhsw9neoUQ8hNePCji1d/x5t84SfagiQUNRVU33V1BqqQh1/12CmvrG5tbxe3Szu7e/kH58KhpkkwLbIhEJbodcINKxtggSQrbqUYeBQpbweh25reeUBuZxA80TtGP+CCWoRScrNTu0hCJ97xeueJW3TnYKvFyUoEc9V75q9tPRBZhTEJxYzqem5I/4ZqkUDgtdTODKRcjPsCOpTGP0PiT+b1TdmaVPgsTbSsmNld/T0x4ZMw4CmxnxGlolr2Z+J/XySi89icyTjPCWCwWhZlilLDZ86wvNQpSY0u40NLeysSQay7IRlSyIXjLL6+S5kXVc6ve/WWldpPHUYQTOIVz8OAKanAHdWiAAAXP8ApvzqPz4rw7H4vWgpPPHMMfOJ8/zWePzA==</latexit><latexit sha1_base64="SrHi+Al9vauGKXsHgyllQXQrrvk=">AAAB73icbVDLSgNBEOyNrxhfUY9eBoPgKeyKoMegF48RzAOSJcxOepMhsw9neoUQ8hNePCji1d/x5t84SfagiQUNRVU33V1BqqQh1/12CmvrG5tbxe3Szu7e/kH58KhpkkwLbIhEJbodcINKxtggSQrbqUYeBQpbweh25reeUBuZxA80TtGP+CCWoRScrNTu0hCJ97xeueJW3TnYKvFyUoEc9V75q9tPRBZhTEJxYzqem5I/4ZqkUDgtdTODKRcjPsCOpTGP0PiT+b1TdmaVPgsTbSsmNld/T0x4ZMw4CmxnxGlolr2Z+J/XySi89icyTjPCWCwWhZlilLDZ86wvNQpSY0u40NLeysSQay7IRlSyIXjLL6+S5kXVc6ve/WWldpPHUYQTOIVz8OAKanAHdWiAAAXP8ApvzqPz4rw7H4vWgpPPHMMfOJ8/zWePzA==</latexit>✓2<latexit sha1_base64="7/pSNJ56+9MBYfOPbWbW9A9Ekos=">AAAB73icbVDLSgNBEOz1GeMr6tHLYBA8hd0g6DHoxWME84BkCbOTTjJk9uFMrxCW/IQXD4p49Xe8+TdOkj1oYkFDUdVNd1eQKGnIdb+dtfWNza3twk5xd2//4LB0dNw0caoFNkSsYt0OuEElI2yQJIXtRCMPA4WtYHw781tPqI2MoweaJOiHfBjJgRScrNTu0giJ96q9UtmtuHOwVeLlpAw56r3SV7cfizTEiITixnQ8NyE/45qkUDgtdlODCRdjPsSOpREP0fjZ/N4pO7dKnw1ibSsiNld/T2Q8NGYSBrYz5DQyy95M/M/rpDS49jMZJSlhJBaLBqliFLPZ86wvNQpSE0u40NLeysSIay7IRlS0IXjLL6+SZrXiuRXv/rJcu8njKMApnMEFeHAFNbiDOjRAgIJneIU359F5cd6dj0XrmpPPnMAfOJ8/zuuPzQ==</latexit><latexit sha1_base64="7/pSNJ56+9MBYfOPbWbW9A9Ekos=">AAAB73icbVDLSgNBEOz1GeMr6tHLYBA8hd0g6DHoxWME84BkCbOTTjJk9uFMrxCW/IQXD4p49Xe8+TdOkj1oYkFDUdVNd1eQKGnIdb+dtfWNza3twk5xd2//4LB0dNw0caoFNkSsYt0OuEElI2yQJIXtRCMPA4WtYHw781tPqI2MoweaJOiHfBjJgRScrNTu0giJ96q9UtmtuHOwVeLlpAw56r3SV7cfizTEiITixnQ8NyE/45qkUDgtdlODCRdjPsSOpREP0fjZ/N4pO7dKnw1ibSsiNld/T2Q8NGYSBrYz5DQyy95M/M/rpDS49jMZJSlhJBaLBqliFLPZ86wvNQpSE0u40NLeysSIay7IRlS0IXjLL6+SZrXiuRXv/rJcu8njKMApnMEFeHAFNbiDOjRAgIJneIU359F5cd6dj0XrmpPPnMAfOJ8/zuuPzQ==</latexit><latexit sha1_base64="7/pSNJ56+9MBYfOPbWbW9A9Ekos=">AAAB73icbVDLSgNBEOz1GeMr6tHLYBA8hd0g6DHoxWME84BkCbOTTjJk9uFMrxCW/IQXD4p49Xe8+TdOkj1oYkFDUdVNd1eQKGnIdb+dtfWNza3twk5xd2//4LB0dNw0caoFNkSsYt0OuEElI2yQJIXtRCMPA4WtYHw781tPqI2MoweaJOiHfBjJgRScrNTu0giJ96q9UtmtuHOwVeLlpAw56r3SV7cfizTEiITixnQ8NyE/45qkUDgtdlODCRdjPsSOpREP0fjZ/N4pO7dKnw1ibSsiNld/T2Q8NGYSBrYz5DQyy95M/M/rpDS49jMZJSlhJBaLBqliFLPZ86wvNQpSE0u40NLeysSIay7IRlS0IXjLL6+SZrXiuRXv/rJcu8njKMApnMEFeHAFNbiDOjRAgIJneIU359F5cd6dj0XrmpPPnMAfOJ8/zuuPzQ==</latexit><latexit sha1_base64="7/pSNJ56+9MBYfOPbWbW9A9Ekos=">AAAB73icbVDLSgNBEOz1GeMr6tHLYBA8hd0g6DHoxWME84BkCbOTTjJk9uFMrxCW/IQXD4p49Xe8+TdOkj1oYkFDUdVNd1eQKGnIdb+dtfWNza3twk5xd2//4LB0dNw0caoFNkSsYt0OuEElI2yQJIXtRCMPA4WtYHw781tPqI2MoweaJOiHfBjJgRScrNTu0giJ96q9UtmtuHOwVeLlpAw56r3SV7cfizTEiITixnQ8NyE/45qkUDgtdlODCRdjPsSOpREP0fjZ/N4pO7dKnw1ibSsiNld/T2Q8NGYSBrYz5DQyy95M/M/rpDS49jMZJSlhJBaLBqliFLPZ86wvNQpSE0u40NLeysSIay7IRlS0IXjLL6+SZrXiuRXv/rJcu8njKMApnMEFeHAFNbiDOjRAgIJneIU359F5cd6dj0XrmpPPnMAfOJ8/zuuPzQ==</latexit>Average loss surfaceLoss surface for data ALoss surface for data B+Published as a conference paper at ICLR 2021

We formalize a notion of consistency, which characterizes to what extent a minimum of the loss
surface appears only when data from different environments are pooled. Minima with low consistency
are ‘patchwork’ solutions, which (we hypothesize) sew together different strategies and should not be
expected to generalize to new environments. An intuitive description of this principle was proposed
by physicist David Deutsch: “good explanations are hard to vary” (Deutsch, 2011).

Using the notion of consistency, we deﬁne Invariant Learning Consistency (ILC), a measure of the
expected consistency of the solution found by a learning algorithm on a given hypothesis class. The
ILC can be improved by changing the hypothesis class or the learning algorithm, and in the last
part of the paper we focus on the latter. We then analyse why current practices in deep learning
provide little incentive for networks to learn invariances, and show that standard training is instead
set up with the explicit objective of greedily maximizing speed of learning, i.e., progress on the
training loss. When learning “as fast as possible” is not the main objective, we show we can trade-off
some “learning speed” for prioritizing learning the invariances. A practical instantiation of ILC leads
to o.o.d. generalization on a challenging synthetic task where several established regularizers fail
to generalize; moreover, following the memorization task from Zhang et al. (2017), ILC prevents
convergence on CIFAR-10 with random labels, as no shared mechanism is present, and similarly
when a portion of training labels is incorrect. Lastly, we set up a behavioural cloning task based on
the game CoinRun (Cobbe et al., 2019b), and observe better generalization on new unseen levels.

An example. Take these two second-hand books
of chess puzzles. We can learn the two independent
shortcuts (blue arrows for the left book OR hand-
written solutions on the right), or actually learn
to play chess (the invariant mechanism). While
both strategies solve other problems from the same
books (i.i.d.), only the latter generalises to new
chess puzzle books (o.o.d.). How to distinguish
the two? We would not have learned about the red
arrows had we trained on the book on the right, and vice versa with the hand-written notes.

2 EXPLANATIONS THAT ARE HARD TO VARY

E

, with |E| “ d, and De “ pxe

i P X Ď Rm
i q, ie “ 1, . . . , ne. Here xe
We consider datasets tDeueP
i P Y Ď Rp the targets. The superscript e P E
is the vector containing the observed inputs, and ye
indexes some aspect of the data collection process, and can be interpreted as an environment label.
Our objective is to infer a function f : X Ñ Y — which we call mechanism — assigning a target ye
i
to each input xe
i ; as explained in the introduction, we assume that such function is shared across all
environments. For estimation purposes, f may be parametrized by a neural network with continuous
activations; for weights θ P Θ Ď Rn, we denote the neural network output at x P X as fθpxq.

i , ye

Gradient-based optimization. To ﬁnd an appropriate model fθ, standard optimizers rely on gradi-
ents from a pooled loss function L : Rn Ñ R. This function measures the average performance of
the neural network when predicting data labels, across all environments: Lpθq :“ 1
Lepθq,
|
E
i q; where (cid:96) : Rp ˆ Rp Ñ r0, `8q is usually chosen to
with Lepθq :“ 1
i ; θq, ye
e|
|
D
be the L2 loss or the cross-entropy loss. The parameter updates according to gradient descent (GD)
are given by θk`1
GD “ θk
GDq, where η ą 0 is the learning rate. Under some standard
assumptions (Lee et al., 2016), pθk
GDqkě0 converges to a local minimizer of L, with probability one.

GD ´ η∇Lpθk

e (cid:96)pf pxe

i ,ye

i qP

ř

ř

pxe

eP

D

E

|

When do we not learn invariances? We start by describing what might prevent learning invari-
ances in standard gradient-based optimization.

(i) Training stops once the loss is low enough. If optimization learned spurious patterns by the time
it converged, invariances will not be learned anymore. This depends on the rate at which different
patterns are learned. The rates at which invariant patterns emerge (and vice-versa, the spurious
patterns do not) can be improved by e.g.: (a) careful architecture design, e.g. as done by hardcoding
spatial equivariance in convolutional networks; (b) ﬁne-tuning models pre-trained on large amounts
of data, where strong features already emerged and can be readily selected.

2

5334Problems,Combinations&Games1.1Matein129580Z0Z0Z0Z7Z0Z0ZpZ06pZ0Z0ZpZ5jpZ0ZPZ040Z0ZnZPA3OPZ0Z0Z020ZKZ0Z0Z1Z0Z0Z0Z0abcdefgh29680Z0Z0Z0Z7snZ0Z0o060j0Z0ZPZ5o0ZPZ0Z04Po0Z0M0Z3ZKZ0Z0Z020ZRZ0Z0Z1Z0Z0Z0Z0abcdefgh29780Z0Z0Z0Z7Z0Zns0Z06BZ0ako0Z5Z0Z0o0Z040O0Z0ZPZ3Z0Z0Z0Z020ZKZNZ0Z1Z0Z0Z0ZRabcdefgh29880Z0Z0Z0Z7ZRZ0Z0Z060Z0ako0Z5Z0Z0onZ040Z0oKZPZ3Z0ZPZ0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh2998rZ0l0j0s7opZ0a0op60Z0o0m0Z5Z0Z0o0M040Z0ZPZbZ3OQM0A0Z020O0Z0OPO1Z0ZRS0J0abcdefgh30080ZrZ0j0s7obl0opZp60onZ0M0Z5Z0m0Z0A040Z0o0Z0Z3ZPZPZPO02PZ0MQZ0O1ZKZ0S0ZRabcdefgh5334Problems,Combinations&Games1.1Matein13018rZbZkZ0s7ZpopZ0Zp6pZ0ZpL0Z5ZNO0Z0Z040OKZ0Z0Z3O0Z0OnZP20Z0Z0O0Z1Z0Z0Z0Z0abcdefgh30280Z0ZkZ0Z7Z0Z0OpZ060Z0O0Z0Z5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0ZQZ020J0Z0Z0Z1Z0Z0l0Z0abcdefgh30380Z0ZkZ0Z7Z0Z0Z0Z060Z0Z0ONZ5Z0ZQZ0Z040Z0Z0Z0Z3Z0Z0ZpZ020Z0ZnZ0J1Z0Z0ZqZ0abcdefgh30480Z0Z0s0Z7j0Z0Z0Z06No0Z0Z0Z5Z0Z0Z0Zp40O0ZbZpO3Z0Z0OrZ02RZ0Z0OKA1Z0Z0Z0Z0abcdefgh30580Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0ZNZ020orZ0ZPO1s0j0J0ZRabcdefgh3068ra0Z0Z0Z7j0o0Z0ZR6PZPZ0Z0Z5OpJ0Z0Z040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0ZBZ0abcdefgh5334Problems,Combinations&Games2.1WhitetoMove#24218rZkZNZ0Z7oRZRZ0Z06KZ0Z0Z0Z5Z0Z0ZnZ040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42280Z0Z0Z0Z7ZNZ0Z0Z06RZpZ0Z0Z5ZkZ0Z0Z040MpZ0Z0Z3Z0J0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42380Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0ZNZB40Z0Z0Z0Z3Z0Z0o0ok20Z0ZRZ0Z1Z0Z0Z0J0abcdefgh42480Z0Z0Z0Z7Z0o0Z0S060ZRZNZ0j5Z0Z0Z0Z040Z0Z0Z0o3Z0Z0Z0ZK20Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42580Z0Z0Z0Z7Z0Z0Z0Zp60Z0Z0Z0L5Z0Z0Z0Z040Z0ZKZko3Z0Z0Z0Z020Z0Z0ZPZ1Z0Z0ZNZ0abcdefgh42680Z0Z0Z0Z7Z0ZRZ0Z060o0Z0Z0Z5ZkZpZ0Z040Z0O0Z0Z3Z0J0Z0Z02QZ0Z0Z0Z1Z0Z0Z0Z0abcdefgh5334Problems,Combinations&Games2.1WhitetoMove#235580Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0ZQZ040Z0Z0mBZ3Z0A0j0Z020Z0Z0Z0Z1Z0Z0ZKZ0abcdefgh35680Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0Z0Z040Z0o0ZNZ3Z0ZKZ0Z020Z0Z0ZpL1Z0Z0ZkZ0abcdefgh35780Z0Z0ZRZ7Z0Z0ZKm060Z0Z0Z0Z5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0Z0S020Z0Z0Z0Z1Z0Z0Z0Akabcdefgh35880Z0ZKZ0Z7Z0Z0Z0Z060ZpZkZ0Z5Z0Z0O0Z040Z0OQZ0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh35980Z0Z0Z0Z7jPO0Z0Z060SnZ0Z0Z5Z0J0Z0Z040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh36080Z0Z0ArZ7Z0Z0Z0O060Z0Z0JBj5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh5334Problems,Combinations&Games2.1WhitetoMove#24218rZkZNZ0Z7oRZRZ0Z06KZ0Z0Z0Z5Z0Z0ZnZ040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42280Z0Z0Z0Z7ZNZ0Z0Z06RZpZ0Z0Z5ZkZ0Z0Z040MpZ0Z0Z3Z0J0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42380Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0ZNZB40Z0Z0Z0Z3Z0Z0o0ok20Z0ZRZ0Z1Z0Z0Z0J0abcdefgh42480Z0Z0Z0Z7Z0o0Z0S060ZRZNZ0j5Z0Z0Z0Z040Z0Z0Z0o3Z0Z0Z0ZK20Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42580Z0Z0Z0Z7Z0Z0Z0Zp60Z0Z0Z0L5Z0Z0Z0Z040Z0ZKZko3Z0Z0Z0Z020Z0Z0ZPZ1Z0Z0ZNZ0abcdefgh42680Z0Z0Z0Z7Z0ZRZ0Z060o0Z0Z0Z5ZkZpZ0Z040Z0O0Z0Z3Z0J0Z0Z02QZ0Z0Z0Z1Z0Z0Z0Z0abcdefghRc7Qc55334Problems,Combinations&Games1.1Matein129580Z0Z0Z0Z7Z0Z0ZpZ06pZ0Z0ZpZ5jpZ0ZPZ040Z0ZnZPA3OPZ0Z0Z020ZKZ0Z0Z1Z0Z0Z0Z0abcdefgh29680Z0Z0Z0Z7snZ0Z0o060j0Z0ZPZ5o0ZPZ0Z04Po0Z0M0Z3ZKZ0Z0Z020ZRZ0Z0Z1Z0Z0Z0Z0abcdefgh29780Z0Z0Z0Z7Z0Zns0Z06BZ0ako0Z5Z0Z0o0Z040O0Z0ZPZ3Z0Z0Z0Z020ZKZNZ0Z1Z0Z0Z0ZRabcdefgh29880Z0Z0Z0Z7ZRZ0Z0Z060Z0ako0Z5Z0Z0onZ040Z0oKZPZ3Z0ZPZ0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh2998rZ0l0j0s7opZ0a0op60Z0o0m0Z5Z0Z0o0M040Z0ZPZbZ3OQM0A0Z020O0Z0OPO1Z0ZRS0J0abcdefgh30080ZrZ0j0s7obl0opZp60onZ0M0Z5Z0m0Z0A040Z0o0Z0Z3ZPZPZPO02PZ0MQZ0O1ZKZ0S0ZRabcdefgh5334Problems,Combinations&Games1.1Matein13018rZbZkZ0s7ZpopZ0Zp6pZ0ZpL0Z5ZNO0Z0Z040OKZ0Z0Z3O0Z0OnZP20Z0Z0O0Z1Z0Z0Z0Z0abcdefgh30280Z0ZkZ0Z7Z0Z0OpZ060Z0O0Z0Z5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0ZQZ020J0Z0Z0Z1Z0Z0l0Z0abcdefgh30380Z0ZkZ0Z7Z0Z0Z0Z060Z0Z0ONZ5Z0ZQZ0Z040Z0Z0Z0Z3Z0Z0ZpZ020Z0ZnZ0J1Z0Z0ZqZ0abcdefgh30480Z0Z0s0Z7j0Z0Z0Z06No0Z0Z0Z5Z0Z0Z0Zp40O0ZbZpO3Z0Z0OrZ02RZ0Z0OKA1Z0Z0Z0Z0abcdefgh30580Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0ZNZ020orZ0ZPO1s0j0J0ZRabcdefgh3068ra0Z0Z0Z7j0o0Z0ZR6PZPZ0Z0Z5OpJ0Z0Z040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0ZBZ0abcdefgh5334Problems,Combinations&Games2.1WhitetoMove#235580Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0ZQZ040Z0Z0mBZ3Z0A0j0Z020Z0Z0Z0Z1Z0Z0ZKZ0abcdefgh35680Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0Z0Z040Z0o0ZNZ3Z0ZKZ0Z020Z0Z0ZpL1Z0Z0ZkZ0abcdefgh35780Z0Z0ZRZ7Z0Z0ZKm060Z0Z0Z0Z5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0Z0S020Z0Z0Z0Z1Z0Z0Z0Akabcdefgh35880Z0ZKZ0Z7Z0Z0Z0Z060ZpZkZ0Z5Z0Z0O0Z040Z0OQZ0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh35980Z0Z0Z0Z7jPO0Z0Z060SnZ0Z0Z5Z0J0Z0Z040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh36080Z0Z0ArZ7Z0Z0Z0O060Z0Z0JBj5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefghKg25334Problems,Combinations&Games1.1Matein13018rZbZkZ0s7ZpopZ0Zp6pZ0ZpL0Z5ZNO0Z0Z040OKZ0Z0Z3O0Z0OnZP20Z0Z0O0Z1Z0Z0Z0Z0abcdefgh30280Z0ZkZ0Z7Z0Z0OpZ060Z0O0Z0Z5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0ZQZ020J0Z0Z0Z1Z0Z0l0Z0abcdefgh30380Z0ZkZ0Z7Z0Z0Z0Z060Z0Z0ONZ5Z0ZQZ0Z040Z0Z0Z0Z3Z0Z0ZpZ020Z0ZnZ0J1Z0Z0ZqZ0abcdefgh30480Z0Z0s0Z7j0Z0Z0Z06No0Z0Z0Z5Z0Z0Z0Zp40O0ZbZpO3Z0Z0OrZ02RZ0Z0OKA1Z0Z0Z0Z0abcdefgh30580Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0Z0Z040Z0Z0Z0Z3Z0Z0ZNZ020orZ0ZPO1s0j0J0ZRabcdefgh3068ra0Z0Z0Z7j0o0Z0ZR6PZPZ0Z0Z5OpJ0Z0Z040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0ZBZ0abcdefghBf85334Problems,Combinations&Games2.1WhitetoMove#24218rZkZNZ0Z7oRZRZ0Z06KZ0Z0Z0Z5Z0Z0ZnZ040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42280Z0Z0Z0Z7ZNZ0Z0Z06RZpZ0Z0Z5ZkZ0Z0Z040MpZ0Z0Z3Z0J0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42380Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0ZNZB40Z0Z0Z0Z3Z0Z0o0ok20Z0ZRZ0Z1Z0Z0Z0J0abcdefgh42480Z0Z0Z0Z7Z0o0Z0S060ZRZNZ0j5Z0Z0Z0Z040Z0Z0Z0o3Z0Z0Z0ZK20Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42580Z0Z0Z0Z7Z0Z0Z0Zp60Z0Z0Z0L5Z0Z0Z0Z040Z0ZKZko3Z0Z0Z0Z020Z0Z0ZPZ1Z0Z0ZNZ0abcdefgh42680Z0Z0Z0Z7Z0ZRZ0Z060o0Z0Z0Z5ZkZpZ0Z040Z0O0Z0Z3Z0J0Z0Z02QZ0Z0Z0Z1Z0Z0Z0Z0abcdefgh5334Problems,Combinations&Games2.1WhitetoMove#24218rZkZNZ0Z7oRZRZ0Z06KZ0Z0Z0Z5Z0Z0ZnZ040Z0Z0Z0Z3Z0Z0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42280Z0Z0Z0Z7ZNZ0Z0Z06RZpZ0Z0Z5ZkZ0Z0Z040MpZ0Z0Z3Z0J0Z0Z020Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42380Z0Z0Z0Z7Z0Z0Z0Z060Z0Z0Z0Z5Z0Z0ZNZB40Z0Z0Z0Z3Z0Z0o0ok20Z0ZRZ0Z1Z0Z0Z0J0abcdefgh42480Z0Z0Z0Z7Z0o0Z0S060ZRZNZ0j5Z0Z0Z0Z040Z0Z0Z0o3Z0Z0Z0ZK20Z0Z0Z0Z1Z0Z0Z0Z0abcdefgh42580Z0Z0Z0Z7Z0Z0Z0Zp60Z0Z0Z0L5Z0Z0Z0Z040Z0ZKZko3Z0Z0Z0Z020Z0Z0ZPZ1Z0Z0ZNZ0abcdefgh42680Z0Z0Z0Z7Z0ZRZ0Z060o0Z0Z0Z5ZkZpZ0Z040Z0O0Z0Z3Z0J0Z0Z02QZ0Z0Z0Z1Z0Z0Z0Z0abcdefghPublished as a conference paper at ICLR 2021

(ii) Learning signals: everything looks relevant for a dataset of size 1. Due to the summation in the
deﬁnition of the pooled loss L, gradients for each example are computed independently. Informally,
each signal is identical to the one for an equivalent dataset of size 1, where every pattern appears
relevant to the task. To ﬁnd invariant patterns across examples, if we compute our training signals on
each of them independently, we have to rely on the way these are aggregated.1

(iii) Aggregating gradients: averaging maximizes learning speed. The default method to pool gradi-
ents is the arithmetic mean. GD applied to L is designed to minimize the pooled loss by prioritizing
descent speed.2 Indeed, a step of GD is equivalent to ﬁnding a tight3 quadratic upper bound ˆL to L,
and then jumping to the minimizer of this approximation (Nocedal and Wright, 2006). While speed is
often desirable, by construction GD ignores one potentially crucial piece of information: The gradient
∇L is the result of averaging signals ∇Le, which correspond to the patterns visible from each environ-
ment at this stage of optimization. In other words, GD with average gradients greedily maximizes for
learning speed, but in some situations we would like to trade some convergence speed for invariance.
For instance, instead of performing an arithmetic mean between gradi-
ents (logical OR), we might want to look towards a logical AND, which
can be characterized as a geometric mean. Fig. 1 shows how a sum can be
seen as a logical OR: the two orthogonal gradients from data A and data B
at (0.5,0.5) point to different directions, yet both are kept in the combined
gradient.4 In Sec. 2.3 we elaborate on this idea and on implementing a
logical AND between gradients. Before presenting this discussion, we take
some time to better motivate the need for invariant learning consistency
and to construct a precise mathematical deﬁnition of consistency.

Figure 2: Inconsistency in
gradient directions.

2.1 FORMAL DEFINITION OF ILC

Let Θ˚
be the set of convergence points of algorithm A when trained using all environments (pooled
“ tθ˚ P Θ | D θ0 P Rn s.t. A8pθ0, Eq “ θ˚u. For instance, if A is gradient
A
data): that is, Θ˚
descent, the result of Lee et al. (2016) implies that Θ˚
A
is the set of local minimizers of the pooled
loss L. To each θ˚ P Θ˚
A
, we want to associate a consistency score, quantifying the concept “good
θ˚ are hard to vary”. In other words, we would like the score to capture the consistency of the loss
A
landscape around θ˚ across the different environments. For example, in Fig. 1 the loss landscape near
the bottom-left minimizer is consistent across environments, while the top-right minimizer is not.
Let us characterize the landscape around θ˚ from the per-
spective of a ﬁxed environment e P E. We deﬁne the set
N (cid:15)
e,θ˚ to be the largest path-connected region of space con-
taining both θ˚ and the set tθ P Θ s.t.|Lepθq ´ Lepθ˚q| ď
(cid:15) u, with (cid:15) ą 0. In other words, if θ P N (cid:15)
e,θ˚ then there
exist a path-connected region in parameter space including
θ˚ and θ where each parameter also is in N (cid:15)
e,θ˚ and its loss
on environment e is comparable. From the perspective of
environment e, all these points are equivalent to θ˚. We would like to evaluate the elements of this
set with respect to a different environment e1 ‰ e. We will say that e1 is consistent with e in θ˚ if
ˇ
ˇ is small. Repeating this reasoning for all environment pairs, we arrive
e,θ˚ |Le1pθq ´ Lepθq
maxθPN (cid:15)
at the following inconsistency score:

I (cid:15)pθ˚q :“ max
pe,e1qP
E

2

max
e,θ˚

θPN (cid:15)

|Le1pθq ´ Lepθ˚q|.

(1)

1After computing the gradients for a dataset of n ´ 1 examples, if an n-th example appeared, we would
just compute one more vector of gradients and add it to the sum. A Gaussian Process (Rasmussen, 2003) for
example would require recomputing the entire solution from scratch, as all interactions are considered.

2The same reasoning holds for SGD in the ﬁnite-sum optimization case L “ 1
m

m
i“1 Li, where gradients

from a mini-batch are seen as unbiased estimators of gradients from the pooled loss. (Bottou et al., 2018).

ř

3Assume that L has L-Lipschitz gradients (i.e. curvature bounded from above by L). Then, at any point ˜θ,

we can construct the upper bound ˆL ˜θpθq “ Lp˜θq ` ∇Lp˜θqJpθ ´ ˜θq ` L}θ ´ ˜θ}2{2.

4Loosely speaking, a sum is large if any of the summands is large, a product is large if all factors are large.

3

θkGDθk+1GDLoss surface for data ALoss surface for data B{✓⇤<latexit sha1_base64="nMxvkX5ox3IwM1Zd//yIbmSI7pU=">AAACGnicdVDLSgMxFL1T3/VVdekmKIKolJm2YLsT3bhUsFrojCWTpm1o5kFyRyhDv8ONC3/EjQtF3Ikb/8ZMa0FFDwQO55yb3Bw/lkKjbX9Yuanpmdm5+YX84tLyymphbf1SR4livM4iGamGTzWXIuR1FCh5I1acBr7kV37/JPOvbrjSIgovcBBzL6DdUHQEo2ikVsFJ3dElTdX1vdQu1g6rpVrtICN2uVbOSLVSKleGLvY40uu9YauwPYmRSYxMYsQp2iNsH225+/cAcNYqvLntiCUBD5FJqnXTsWP0UqpQMMmHeTfRPKasT7u8aWhIA669dLTVkOwYpU06kTInRDJSv0+kNNB6EPgmGVDs6d9eJv7lNRPsVL1UhHGCPGTjhzqJJBiRrCfSFoozlANDKFPC7EpYjyrK0LSZNyVMfkr+J5elomMXnXPTxjGMMQ+bsAW74MAhHMEpnEEdGNzCAzzBs3VnPVov1us4mrO+ZjbgB6z3T807nxY=</latexit><latexit sha1_base64="COHTBSE8Do/ylWnLOwFUtqtv35c=">AAACGnicdVDLSgMxFM34rPVVdekmtAiiUmbaQttd0Y3LCvYBnbFk0rQNzTxI7ghl6F8IbvwVNy4UcSdu+jdmWgsqeiBwOOfc5Oa4oeAKTHNqLC2vrK6tpzbSm1vbO7uZvf2mCiJJWYMGIpBtlygmuM8awEGwdigZ8VzBWu7oIvFbt0wqHvjXMA6Z45GBz/ucEtBSN2PF9uySjhy4Tmzmq+VKoVo9S4hZrBYTUikViqWJDUMG5OZk0s3kFjG8iOFFDFt5c4ZcLWuf3k1r43o38273Ahp5zAcqiFIdywzBiYkETgWbpO1IsZDQERmwjqY+8Zhy4tlWE3yklR7uB1IfH/BM/T4RE0+psefqpEdgqH57ifiX14mgX3Fi7ocRMJ/OH+pHAkOAk55wj0tGQYw1IVRyvSumQyIJBd1mWpew+Cn+nzQLecvMW1e6jXM0Rwodoiw6RhYqoxq6RHXUQBTdo0f0jF6MB+PJeDXe5tEl42vmAP2A8fEJ1qWgnA==</latexit><latexit sha1_base64="COHTBSE8Do/ylWnLOwFUtqtv35c=">AAACGnicdVDLSgMxFM34rPVVdekmtAiiUmbaQttd0Y3LCvYBnbFk0rQNzTxI7ghl6F8IbvwVNy4UcSdu+jdmWgsqeiBwOOfc5Oa4oeAKTHNqLC2vrK6tpzbSm1vbO7uZvf2mCiJJWYMGIpBtlygmuM8awEGwdigZ8VzBWu7oIvFbt0wqHvjXMA6Z45GBz/ucEtBSN2PF9uySjhy4Tmzmq+VKoVo9S4hZrBYTUikViqWJDUMG5OZk0s3kFjG8iOFFDFt5c4ZcLWuf3k1r43o38273Ahp5zAcqiFIdywzBiYkETgWbpO1IsZDQERmwjqY+8Zhy4tlWE3yklR7uB1IfH/BM/T4RE0+psefqpEdgqH57ifiX14mgX3Fi7ocRMJ/OH+pHAkOAk55wj0tGQYw1IVRyvSumQyIJBd1mWpew+Cn+nzQLecvMW1e6jXM0Rwodoiw6RhYqoxq6RHXUQBTdo0f0jF6MB+PJeDXe5tEl42vmAP2A8fEJ1qWgnA==</latexit><latexit sha1_base64="1Lh+CLV6bLWCOAmXYGPxhAOxZH0=">AAACGnicdVDLSgMxFM3UV62vqks3wSKISJlpC213RTcuK9gHtGPJpGkbmnmQ3BHKMN/hxl9x40IRd+LGvzEzbUFFDwQO55yb3BwnEFyBaX4amZXVtfWN7GZua3tndy+/f9BWfigpa1Ff+LLrEMUE91gLOAjWDSQjriNYx5leJn7njknFfe8GZgGzXTL2+IhTAloa5K2on17Sk2PHjsxivVor1evnCTHL9XJCapVSuRL3YcKA3J7Fg3xhGcPLGF7GsFU0UxTQAs1B/r0/9GnoMg+oIEr1LDMAOyISOBUszvVDxQJCp2TMepp6xGXKjtKtYnyilSEe+VIfD3Cqfp+IiKvUzHV00iUwUb+9RPzL64UwqtkR94IQmEfnD41CgcHHSU94yCWjIGaaECq53hXTCZGEgm4zp0tY/hT/T9qlomUWrWuz0LhY1JFFR+gYnSILVVEDXaEmaiGK7tEjekYvxoPxZLwab/NoxljMHKIfMD6+ALq8nY0=</latexit>✓⇤<latexit sha1_base64="nMxvkX5ox3IwM1Zd//yIbmSI7pU=">AAACGnicdVDLSgMxFL1T3/VVdekmKIKolJm2YLsT3bhUsFrojCWTpm1o5kFyRyhDv8ONC3/EjQtF3Ikb/8ZMa0FFDwQO55yb3Bw/lkKjbX9Yuanpmdm5+YX84tLyymphbf1SR4livM4iGamGTzWXIuR1FCh5I1acBr7kV37/JPOvbrjSIgovcBBzL6DdUHQEo2ikVsFJ3dElTdX1vdQu1g6rpVrtICN2uVbOSLVSKleGLvY40uu9YauwPYmRSYxMYsQp2iNsH225+/cAcNYqvLntiCUBD5FJqnXTsWP0UqpQMMmHeTfRPKasT7u8aWhIA669dLTVkOwYpU06kTInRDJSv0+kNNB6EPgmGVDs6d9eJv7lNRPsVL1UhHGCPGTjhzqJJBiRrCfSFoozlANDKFPC7EpYjyrK0LSZNyVMfkr+J5elomMXnXPTxjGMMQ+bsAW74MAhHMEpnEEdGNzCAzzBs3VnPVov1us4mrO+ZjbgB6z3T807nxY=</latexit><latexit sha1_base64="COHTBSE8Do/ylWnLOwFUtqtv35c=">AAACGnicdVDLSgMxFM34rPVVdekmtAiiUmbaQttd0Y3LCvYBnbFk0rQNzTxI7ghl6F8IbvwVNy4UcSdu+jdmWgsqeiBwOOfc5Oa4oeAKTHNqLC2vrK6tpzbSm1vbO7uZvf2mCiJJWYMGIpBtlygmuM8awEGwdigZ8VzBWu7oIvFbt0wqHvjXMA6Z45GBz/ucEtBSN2PF9uySjhy4Tmzmq+VKoVo9S4hZrBYTUikViqWJDUMG5OZk0s3kFjG8iOFFDFt5c4ZcLWuf3k1r43o38273Ahp5zAcqiFIdywzBiYkETgWbpO1IsZDQERmwjqY+8Zhy4tlWE3yklR7uB1IfH/BM/T4RE0+psefqpEdgqH57ifiX14mgX3Fi7ocRMJ/OH+pHAkOAk55wj0tGQYw1IVRyvSumQyIJBd1mWpew+Cn+nzQLecvMW1e6jXM0Rwodoiw6RhYqoxq6RHXUQBTdo0f0jF6MB+PJeDXe5tEl42vmAP2A8fEJ1qWgnA==</latexit><latexit sha1_base64="COHTBSE8Do/ylWnLOwFUtqtv35c=">AAACGnicdVDLSgMxFM34rPVVdekmtAiiUmbaQttd0Y3LCvYBnbFk0rQNzTxI7ghl6F8IbvwVNy4UcSdu+jdmWgsqeiBwOOfc5Oa4oeAKTHNqLC2vrK6tpzbSm1vbO7uZvf2mCiJJWYMGIpBtlygmuM8awEGwdigZ8VzBWu7oIvFbt0wqHvjXMA6Z45GBz/ucEtBSN2PF9uySjhy4Tmzmq+VKoVo9S4hZrBYTUikViqWJDUMG5OZk0s3kFjG8iOFFDFt5c4ZcLWuf3k1r43o38273Ahp5zAcqiFIdywzBiYkETgWbpO1IsZDQERmwjqY+8Zhy4tlWE3yklR7uB1IfH/BM/T4RE0+psefqpEdgqH57ifiX14mgX3Fi7ocRMJ/OH+pHAkOAk55wj0tGQYw1IVRyvSumQyIJBd1mWpew+Cn+nzQLecvMW1e6jXM0Rwodoiw6RhYqoxq6RHXUQBTdo0f0jF6MB+PJeDXe5tEl42vmAP2A8fEJ1qWgnA==</latexit><latexit sha1_base64="1Lh+CLV6bLWCOAmXYGPxhAOxZH0=">AAACGnicdVDLSgMxFM3UV62vqks3wSKISJlpC213RTcuK9gHtGPJpGkbmnmQ3BHKMN/hxl9x40IRd+LGvzEzbUFFDwQO55yb3BwnEFyBaX4amZXVtfWN7GZua3tndy+/f9BWfigpa1Ff+LLrEMUE91gLOAjWDSQjriNYx5leJn7njknFfe8GZgGzXTL2+IhTAloa5K2on17Sk2PHjsxivVor1evnCTHL9XJCapVSuRL3YcKA3J7Fg3xhGcPLGF7GsFU0UxTQAs1B/r0/9GnoMg+oIEr1LDMAOyISOBUszvVDxQJCp2TMepp6xGXKjtKtYnyilSEe+VIfD3Cqfp+IiKvUzHV00iUwUb+9RPzL64UwqtkR94IQmEfnD41CgcHHSU94yCWjIGaaECq53hXTCZGEgm4zp0tY/hT/T9qlomUWrWuz0LhY1JFFR+gYnSILVVEDXaEmaiGK7tEjekYvxoPxZLwab/NoxljMHKIfMD6+ALq8nY0=</latexit>N✏A,✓⇤<latexit sha1_base64="zE2iFOVU4whfETknWj2UJQZnji8=">AAACAnicbVA9SwNBEJ3zM8avqJXYHIogKuHORsuojZVEMImQxLC3mZgle3vH7pwQjqCFP0UbC0Vs/RV2/hB7Nx+FXw8GHu/NMDMviKUw5Hkfztj4xOTUdGYmOzs3v7CYW1oumyjRHEs8kpG+CJhBKRSWSJDEi1gjCwOJlaBz3Pcr16iNiNQ5dWOsh+xKiZbgjKzUyK2eXqY1jI2Qkeo10sPdGrWR2OV2r5Hb8PLeAO5f4o/IRmHn8/4WAIqN3HutGfEkREVcMmOqvhdTPWWaBJfYy9YSgzHjHXaFVUsVC9HU08ELPXfTKk23FWlbityB+n0iZaEx3TCwnSGjtvnt9cX/vGpCrYN6KlScECo+XNRKpEuR28/DbQqNnGTXEsa1sLe6vM0042RTy9oQ/N8v/yXlvbzv5f0zm8YRDJGBNViHLfBhHwpwAkUoAYcbeIAneHbunEfnxXkdto45o5kV+AHn7Qu4B5nd</latexit><latexit sha1_base64="WxNfmbQCHqjHwXB6lzaw2Bpm+zk=">AAACAnicbVC7SgNBFJ31GeNr1UpsFoMgKmHXRsuojZVEMA/Ii9nJTTJkdnaZuSuEJdjY+RvaWChi61fY+SH2Th6FJh64cDjnXu69x48E1+i6X9bM7Nz8wmJqKb28srq2bm9sFnUYKwYFFopQlX2qQXAJBeQooBwpoIEvoOR3LwZ+6RaU5qG8wV4EtYC2JW9xRtFIDXv7qp5UIdJchLLfSM6OqtgBpPWDfsPOuFl3CGeaeGOSyR1+Pz5AJso37M9qM2RxABKZoFpXPDfCWkIVciagn67GGiLKurQNFUMlDUDXkuELfWfPKE2nFSpTEp2h+nsioYHWvcA3nQHFjp70BuJ/XiXG1mkt4TKKESQbLWrFwsHQGeThNLkChqJnCGWKm1sd1qGKMjSppU0I3uTL06R4nPXcrHdt0jgnI6TIDtkl+8QjJyRHLkmeFAgjd+SJvJBX6956tt6s91HrjDWe2SJ/YH38ABIymuA=</latexit><latexit sha1_base64="WxNfmbQCHqjHwXB6lzaw2Bpm+zk=">AAACAnicbVC7SgNBFJ31GeNr1UpsFoMgKmHXRsuojZVEMA/Ii9nJTTJkdnaZuSuEJdjY+RvaWChi61fY+SH2Th6FJh64cDjnXu69x48E1+i6X9bM7Nz8wmJqKb28srq2bm9sFnUYKwYFFopQlX2qQXAJBeQooBwpoIEvoOR3LwZ+6RaU5qG8wV4EtYC2JW9xRtFIDXv7qp5UIdJchLLfSM6OqtgBpPWDfsPOuFl3CGeaeGOSyR1+Pz5AJso37M9qM2RxABKZoFpXPDfCWkIVciagn67GGiLKurQNFUMlDUDXkuELfWfPKE2nFSpTEp2h+nsioYHWvcA3nQHFjp70BuJ/XiXG1mkt4TKKESQbLWrFwsHQGeThNLkChqJnCGWKm1sd1qGKMjSppU0I3uTL06R4nPXcrHdt0jgnI6TIDtkl+8QjJyRHLkmeFAgjd+SJvJBX6956tt6s91HrjDWe2SJ/YH38ABIymuA=</latexit><latexit sha1_base64="UznaIS8ZfVFCNGlNZu2HAS8/RTk=">AAACAnicbVBNS8NAEN3Ur1q/op7ES7AIIlISL3qsevEkFewHNGnZbKft0s0m7E6EEooX/4oXD4p49Vd489+4/Tho9cHA470ZZuaFieAaXffLyi0sLi2v5FcLa+sbm1v29k5Nx6liUGWxiFUjpBoEl1BFjgIaiQIahQLq4eBq7NfvQWkeyzscJhBEtCd5lzOKRmrbezetzIdEcxHLUTu7OPGxD0hbx6O2XXRL7gTOX+LNSJHMUGnbn34nZmkEEpmgWjc9N8Egowo5EzAq+KmGhLIB7UHTUEkj0EE2eWHkHBql43RjZUqiM1F/TmQ00noYhaYzotjX895Y/M9rptg9DzIukxRBsumibiocjJ1xHk6HK2AohoZQpri51WF9qihDk1rBhODNv/yX1E5Lnlvybt1i+XIWR57skwNyRDxyRsrkmlRIlTDyQJ7IC3m1Hq1n6816n7bmrNnMLvkF6+MbUmuXWQ==</latexit>Published as a conference paper at ICLR 2021

This consistency is our formalization of the principle “good explanations are hard to vary”. Finally,
we can write down an invariant learning consistency score for A:

ILCpA, pθ0q :“ ´E

θ0„ppθ0q

I (cid:15)pA8pθ0, Eq

.

(2)

“

‰

That is, the learning consistency of an algorithm measures the expected consistency across environ-
ments of the minimizer it converges to on the pooled data.
Example: low consistency of a classic patchwork solution. One-hidden-layer networks with
sigmoid activations and enough neurons can approximate any function f ˚ : r0, 1s Ñ R (Cybenko,
1989). In appendix A.1 we show how the construction used to obtain the weights leads to a maximally
inconsistent solution according to I (cid:15)pθ˚q, which would not be expected to generalize o.o.d.

ILC AS A LOGICAL AND BETWEEN LANDSCAPES

2.2
Here we draw a connection between our deﬁnition of inconsistency
and the local geometric properties of the loss landscapes. For the sake
of clarity, we consider two environments (A and B) and assume θ˚ to
be a local minimizer (with zero loss) for both environments. Using a
Taylor approximation5, we get Lpθq « 1
2 pθ ´ θ˚qJHA`Bpθ ´ θ˚q for
}θ ´ θ˚} « 0, where HA`B “ pHA ` HBq {2 is the arithmetic mean
of the Hessians HA :“ ∇2LApθ˚q and HB :“ ∇2LApθ˚q. HA`B
does not capture the possibly conﬂicting geometries of landscape A
or B: It performs a “logical OR” on the dominant eigendirections. In
contrast, the geometric mean, or Karcher mean, HA^B (Ando et al.,
2004) is affected by the inconsistencies between landscapes: It performs
a “logical AND”. In appendix A.2, we give a formal deﬁnition of HA^B,
and show that for diagonal Hessians, I (cid:15)pθ˚q ď 2(cid:15)p detpHA`B q
detpHA^B q q2. As
for the geometric mean of positive numbers, 0 ď detpHA^Bq ď detpHA`Bq; thus, inconsistency is
lowest when shapes of A and B are similar – exactly as in the bottom-left minimizer of Fig. 1.

Figure 3: Plotted are con-
tour lines θJH ´1θ “ 1
for HA “ diagp0.05, 1q and
HB “ diagp1, 0.05q. HA^B
retains the original volumes,
while for HA`B it is 5ˆ big-
ger. This magniﬁcation shows
inconsistency of A and B.

E

E

eP

ś

ś

λe
1q1{|

|, . . . , p

i are positive, their geometric mean is H ^ :“ diagpp

From Hessians to gradients. We just saw that the consistency of θ˚ is linked to the geometric mean
. Under the simplifying assumption that each He is diagonal6 and all
of the Hessians tHepθ˚queP
eigenvalues λe
|q.
The curvature of the corresponding loss in the i-th eigendirection depends on how consistent the cur-
vatures of each environment are in that direction. Consider now optimizing from a point θk; gradient
descent reads θk`1 “ θk ´ηH `pθk ´θ˚q, where H ` :“ diagp 1
1, . . . , 1
λe
λe
nq. For η
|
|
|
|
E
E
small enough7, we have |θk`1
i ´ θ˚
i |. As noted, this choice maximises
the speed of convergence to θ˚, but does not take into account whether this minimizer is consistent. We
can reduce the speed of convergence on directions where landscapes have different curvatures – which
would lead to a high inconsistency – by following the gradients from the geometric mean of the land-
scapes, as opposed to the arithmetic mean. I.e, we substitute the full gradient ∇Lpθq “ H `pθk ´ θ˚q
with ∇L^pθq “ H ^pθk ´ θ˚q. Also, we have that8 ∇L^pθq “ p
|: to reduce the
speed of convergence in directions with inconsistency, we can take the element-wise geometric mean
of gradients from different environments (see also Fig. 11 in the appendix).

i | “ p1 ´ η 1
|
E

i ´ θ˚

∇Lepθqq

i q|θk
λe

nq1{|
λe

ś

ř

ř

ř

eP

eP

eP

eP

eP

1{|

E

E

E

E

E

E

E

E

|

2.3 MASKING GRADIENTS WITH A LOGICAL AND
The element-wise geometric mean of gradients, instead of the arithmetic mean, increases consistency
in the convex quadratic case. However, there are a few practical limitations:
(i) The geometric mean is only deﬁned when all the signs are consistent. It is still to be deﬁned how

sign inconsistencies, which can occur in non-convex settings, should be dealt with.

(ii) It provides little ﬂexibility for ‘partial’ agreement: Even a single zero gradient component in one

environment stops optimization in that direction.

5This provides a useful simpliﬁed perspective. Indeed, this quadratic model is heavily used in the optimization

community (see e.g. Jastrz˛ebski et al. (2017); Zhang et al. (2019a); Mandt et al. (2017).)

6It was shown in (Becker et al., 1988) and recently in (Adolphs et al., 2019; Singh and Alistarh, 2020) that

neural networks have a strong diagonal dominance of the Hessian matrix at the end of training.

7Smaller than 1{λmax, λmax is the maximum eigenvalue of Hessians from different environments,
8This holds if θ ´ θ˚ is positive, otherwise we have ∇L^pθq “ ´

|∇Lepθq|

`ś

|.

1{|

˘

E

eP

E

4

-1.5-1-0.500.511.5-1-0.8-0.6-0.4-0.200.20.40.60.81HA,HBHA+BHA∧BPublished as a conference paper at ICLR 2021

(iii) For numerical stability, it needs to be computed in log domain (more computationally expensive).
(iv) Adaptive step-size schemes (e.g. Adam (Kingma and Ba, 2015)) rescale the signal component-
wise for local curvature adaptation. The exact magnitude of the geometric mean would be ignored
and most of the difference from arithmetic averaging will come from the zero-ed components.
(i) can be overcome by treating different signs as zeros, resulting in a geometric mean of 0 if there
is any sign disagreement across environments for a gradient component. For (ii) we can allow for
some disagreement (with a hyperparameter), by not masking out if there is a large percentage of
environments with gradients in that direction. (iii) and (iv) can be addressed together: Since the
ﬁnal magnitude will be rescaled except for masked components, i.e. where the geometric mean is 0,
we can use the average gradients (fast to compute) and mask out the components based on the sign
agreement (computable avoiding the log domain).

The AND-mask. We translate the reasoning we just presented to a practical algorithm that we will
refer to as the AND-mask. In its most simple implementation, we zero out those gradient components
with respect to weights that have inconsistent signs across environments. Formally, the masked
gradients at iteration k are mtpθkq d ∇Lpθkq, where mtpθkq vanishes for any component where there
are less than t P td{2, d{2 ` 1, . . . , du agreeing gradient signs across environments (d is the number
of environments in the batch), and is equal to one otherwise. For convenience, our implementation of
the AND-mask uses a threshold τ P r0, 1s as hyper-parameter instead of t, such that t “ d
2 pτ ` 1q.
Mathematically, for every component rmτ sj of mτ , rmτ sj “ 1 rτ d ď |
e signpr∇Lesjq|s.
Computing the AND-mask has the same time and space complexity of standard gradient descent,
i.e., linear in the number of examples that we average. Due to its simplicity and computational
efﬁciency, this is the algorithm that we will use in the experiment section. As a ﬁrst result, we show
that following the AND-masked gradient leads to convergence in the directions made visible by the
AND-mask. The proof is presented in appendix A.3.

ř

Proposition 1. Let L have L-Lipschitz gradients and consider a learning rate η ď 1{L. After k
iterations, AND-masked GD visits at least once a point θ where }mtpθq d ∇Lpθq}2 ď Op1{kq.

Behaviour in the face of randomness. Here we put the
AND mask through a theoretical test: For gradients coming
from different environments that are inconsistent (or even ran-
dom), how fast does the AND mask reduce the magnitude of
the step taken in parameter space, compared to standard GD?
In case of inconsistency, the AND mask should quickly make
the gradient steps more conservative.

To assess this property, we consider a ﬁxed set of n parameters
θ and gradients ∇Le drawn independently from a multivariate
Gaussian with zero mean and unit covariance.

Figure 4: Magnitude of gradient (aver-
age or masked) on random data (|θ| =
3000, t “ 0.8d).

ř

d

e“1 Le. While E}∇Lpθq}2 “
Proposition 2. Consider the setting we just outlined, with L “ p1{dq
Opn{dq, we have that @t P td{2 ` 1, . . . , du, Dc P p1, 2s such that E}mtpθq d ∇Lpθq}2 ď Opn{cdq.
The proof is presented in Appendix A.4, and an illustration with numerical veriﬁcation in Fig. 4 (the
magnitudes of masked gradients (•) for more than 100 examples were always zero in the numerical
veriﬁcation). Intuitively, in the presence of purely random patterns, the AND-mask has a desirable
property: it decreases the strength of these signals exponentially fast, as opposed to linearly.

3 EXPERIMENTS

Real-world datasets are generated by (causal) generative processes which share mechanisms (Pearl,
2009). However, mechanisms and spurious signals are often entangled, making it hard to assess what
part of the learning signal is due to either. As the goal of this paper is to dissect these two components
to understand how they ultimately contribute to the learning process, we create a simple synthetic
dataset that allows us to control the complexity, intensity, and number of shortcuts in the data. After
that, we evaluate whether spurious signals can be detected even in high-dimensional networks and
datasets by testing the AND-mask on a memorization task similar to the one proposed in Zhang et al.
(2017), and on a behavioral cloning task using the game CoinRun (Cobbe et al., 2019a).

5

101102Number of examples10231015107101Norm of the average gradientsaverage gradientsmasked (upper bound)Published as a conference paper at ICLR 2021

Figure 5: A 4-dimensional instantiation of the synthetic memorization dataset for visualization. Every example
is a dot in both circles, and it can be classiﬁed by ﬁnding either of the “oracle” decision boundaries shown.

3.1 THE SYNTHETIC MEMORIZATION DATASET

We introduce a binary classiﬁcation task. The input dimensionality is d “ dM ` dS. While ppy|xdM q
is the same across all environments (i.e. the mechanism), ppy|xdS , eq is not the same across all
environments (the shortcuts). While the mechanism is shared, it needs a highly non-linear decision
boundary to classify the data. The shortcuts are not shared across environments, but provide a simple
way to classify the data, even when pooling all the environments together. See Figure 5 for a concrete
example with dM and dS equal to 2, and two environments (A and B). The spirals (on dM ) are
invariant but hard to model. The shortcuts (on dS) are simple blobs but different in every environment:
in A, linearly separable through a vertical decision boundary, in B with a horizontal one. If the two
environments are pooled, a new diagonal decision boundary emerges on the shortcut dimensions as
the most ‘natural’ one. While this perfectly classiﬁes data in both environments A and B, critically
it would have not been found by training on either partition A or B alone. The out-of-distribution
(o.o.d.) test data has the same mechanism but random shortcuts. Therefore, any method relying
exclusively on the shortcuts will have chance-level o.o.d. performance. Details about the dataset,
baselines, and training curves are reported in appendix B.

Despite the apparent simplicity of this dataset, note that it is challenging to ﬁnd the invariant
mechanism. In high dimensions, even with tens of pooled environments, the shortcuts allow for a
simple classiﬁcation rule under almost every classical deﬁnition of ‘simple’: the boundary is linear, it
has a large margin, it can be expressed with small weights, it is fast to learn, robust to input noise,
and has perfect accuracy and no i.i.d. generalization gap. Finding the complex decision boundary of
the spirals, instead, is a ﬁddly process and arguably a much slower path towards small loss.

Baselines. We evaluate several domain-agnostic baselines (all multilayer perceptrons) with some
of the most common regularizers used in deep learning — Dropout, L1, L2, Batch normalization.
We also consider methods that explicitly make use of the environment labels, namely: (i) Domain
Adversarial Neural Networks (DANN) (Ganin et al., 2016), a method speciﬁcally designed to address
domain adaptation by obfuscating domain information with an adversarial classiﬁer; (ii) Invariant
Risk Minimization (IRM) (Arjovsky et al., 2019), discussed in detail in appendix B. The AND-mask
is trained with the same conﬁgurations in Table 1.

Results. Fig. 6 shows training and test accuracy.
DANN fails because it can align the representation-
layer distributions from different environments using
only shortcuts, such that they become indistinguishable
to the domain-discriminating classiﬁer. The AND-mask
was the only method to achieve perfect test accuracy,
by ﬁtting the spirals instead of the shortcuts. In par-
ticular, the combination of the AND-mask with L1 or
L2 regularization gave the most robust results overall,
as they help suppress neurons that at initialization are
tuned towards the shortcuts.

Figure 6: Results on the synthetic dataset.

Correlations between average, memorization and generalization gradients. Due to the syn-
thetic nature of the dataset, we can intervene on its data-generating process in order to examine
the learning signals coming from the mechanisms and from the shortcuts. We isolate the two and
measure their contribution to the average gradients, as we vary the agreement threshold of the
mask. More precisely, we look at the gradients computed with respect to the weights of a ran-
domly initialized network for different sets of data: (i) The original data, with mechanisms and
shortcuts. (ii) Randomly permuting the dataset over the mechanisms dimensions, thus leaving the
“memorization” signal of the shortcuts. (iii) Randomly permuting over the shortcuts dimensions,
isolating the “generalization” signal of the mechanisms alone. Figure 7 shows the correlation be-

6

Environment AEnvironment BPooled A & BTest o.o.d.dS<latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit>dM<latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit>dS<latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit>dM<latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit>dS<latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit>dM<latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit>dS<latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit><latexit sha1_base64="ZGdevuGtb+DGYShaxNFHLH7aCAQ=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBgx4rtR/QhrLZbNqlm03YnQgl9Cd48aCIV3+RN/+N2zYHrT4YeLw3w8y8IJXCoOt+OaW19Y3NrfJ2ZWd3b/+genjUMUmmGW+zRCa6F1DDpVC8jQIl76Wa0ziQvBtMbuZ+95FrIxL1gNOU+zEdKREJRtFKrXDYGlZrbt1dgPwlXkFqUKA5rH4OwoRlMVfIJDWm77kp+jnVKJjks8ogMzylbEJHvG+pojE3fr44dUbOrBKSKNG2FJKF+nMip7Ex0ziwnTHFsVn15uJ/Xj/D6NrPhUoz5IotF0WZJJiQ+d8kFJozlFNLKNPC3krYmGrK0KZTsSF4qy//JZ2LuufWvfvLWuO2iKMMJ3AK5+DBFTTgDprQBgYjeIIXeHWk8+y8Oe/L1pJTzBzDLzgf3yJBjbM=</latexit>dM<latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit><latexit sha1_base64="J0iXxF81mAuEgtAgVuQW5AxLU34=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0lE0GPBg16EitYW2lA2m0m7dLMJuxuhhP4ELx4U8eov8ua/cdvmoK0PBh7vzTAzL0gF18Z1v53Syura+kZ5s7K1vbO7V90/eNRJphi2WCIS1QmoRsEltgw3AjupQhoHAtvB6Grqt59QaZ7IBzNO0Y/pQPKIM2qsdB/2b/vVmlt3ZyDLxCtIDQo0+9WvXpiwLEZpmKBadz03NX5OleFM4KTSyzSmlI3oALuWShqj9vPZqRNyYpWQRImyJQ2Zqb8nchprPY4D2xlTM9SL3lT8z+tmJrr0cy7TzKBk80VRJohJyPRvEnKFzIixJZQpbm8lbEgVZcamU7EheIsvL5PHs7rn1r2781rjuoijDEdwDKfgwQU04Aaa0AIGA3iGV3hzhPPivDsf89aSU8wcwh84nz8ZKY2t</latexit>Dropout, L2, L1DANNIRMAND-mask(ours)0.40.50.60.70.80.91.0AccuracyTrainTestPublished as a conference paper at ICLR 2021

tween the components of the original average gradient (i) and the shortcut gradients ((ii), dashed
line), and between the original average gradients and the mechanism gradients ((iii), solid line).
While the signal from the mechanisms is present in the
original average gradients (i.e. ρ « 0.4 for τ “ 0), its mag-
nitude is smaller and it is ‘drowned’ by the memorization
signal. Instead, increasing the threshold of the AND-mask
(right side) suppresses memorization gradients due to the
shortcuts, and for τ « 1 most of the gradient components
remaining contain signal from the mechanism. On the left
side, we test the other side of our hypothesis: An XOR-
mask zeroes out consistent gradients, preserves those with
different signs, and results in a sharper decrease of the
correlation with the mechanism gradients.

Figure 7: Gradient correlations.

3.2 EXPERIMENTS ON CIFAR-10

Memorization in a vision task. Zhang et al. (2017) showed that neural networks trained with
standard regularizers — like L2 and Dropout — can still memorize large training datasets with
shufﬂed labels, i.e. reaching «100% training accuracy. Their experiments raised signiﬁcant questions
about the generalization properties of neural networks and the role of regularizers in constraining
the hypothesis class. Our hypothesis is that ILC — for example implemented as the AND-mask
— should prevent memorization on a similar task with the shufﬂed labels, as gradients will tend to
largely ‘disagree’ in the absence of a shared mechanism. However, when the labels are not shufﬂed,
ILC should have a much weaker effect, as real shared mechanisms are still present in the data.

To test our hypothesis, we ran an experiment that closely resembles the one in (Zhang et al., 2017) on
CIFAR-10. We trained a ResNet on CIFAR-10 with random labels, with and without the AND-mask.
In all experiments we used batch size 80, and treated each example as its own “environment”. Recall
that standard gradient averaging is equivalent to an AND-mask with threshold 0. As shown in Figure
8, the ResNet with standard average gradients memorized the data, while slightly increasing the
threshold for the AND-mask quickly prevented memorization (dark blue line). In contrast, training
the same networks on the dataset with the original labels resulted in both of them converging and
generalizing to the test set, conﬁrming that the mask did not signiﬁcantly affect the generalization
error with a general underlying mechanism in the data.

Note that there is no standard notion of environments in CIFAR-
10, which is why we treated every example as coming from
its own environment. This assumption is not unreasonable,
as every image in the dataset was literally collected in a dif-
ferent physical environment. If anything, it is the standard
i.i.d. assumption that hides this variety behind a notion of a
single distribution encompassing all environments. The results
of this experiment further support this interpretation, and can
serve as evidence that — in some cases — we might be able
to identify invariances even without an explicit partition into
environments, as this can be already identiﬁed at the level of
individual examples.

Label noise. Following up on this experiment, we test how
the AND-mask performs in the presence of label noise, i.e. when
a portion of the labels in the training set are randomly shufﬂed
(25% here). According to our hypothesis, gradients computed
on examples with random labels should disagree and get masked
out by the AND-mask, while signal from correctly labeled data
should contribute to update the model. As shown in Figure 9,
the performance on the incorrectly labeled portion of the dataset
is well below chance for the AND-mask (as it predicts correctly
despite the wrong labels), while the baseline again memorizes
the incorrect labels. On the test set (with untouched labels), the
baseline peaks early then decreases as the model overﬁts, while
the AND-mask slowly but steadily improves.

7

Figure 8: As the AND-mask threshold
increases, memorization on CIFAR-10
with random labels is quickly hindered.

Figure 9: The AND-mask prevents
overﬁtting to the incorrectly labeled
portion of the training set (left) without
hurting the test accuracy (right).

1.00.50.00.51.0Agreementthresholdτ0.250.500.751.00CorrelationcoeﬃcientAND-mask→←XOR-maskcorr(∇Lmech,mτ(∇L))corr(∇Lshortcut,mτ(∇L))      0.00      0.05      0.10      0.20      0.40      0.60      0.80Agreement threshold0.250.500.751.00Training AccuracyAND-maskStandard LabelsRandom Labels080Epoch0.000.250.500.751.00Train accuracy on mislabeled portionRandom080EpochTest accuracyRandomAdamAND-maskPublished as a conference paper at ICLR 2021

3.3 BEHAVIORAL CLONING ON COINRUN

CoinRun (Cobbe et al., 2019b) is a game introduced to test how RL agents generalize to novel
situations. The agent needs to collect coins, jumping on top of walls and boxes and avoiding
enemies.9 Each level is procedurally generated — i.e. it has a different combination of sprites,
background, and layout — but the physics and goals are invariant. Cobbe et al. (2019b) showed that
state-of-the-art RL algorithms fail to model these invariant mechanisms, performing poorly on new
levels unless trained on thousands of them. To test our hypothesis, we set up a behavioral cloning
task using CoinRun.10 We start by pre-training a strong policy π˚ using standard PPO (Schulman
et al., 2017) for 400M steps on the full distribution of levels. We then generate a dataset of pairs
ps, π˚pa|sqq from the on-policy distribution. The training data consists of 1000 states from each
of 64 levels, while test data comes from 2000 levels. A ResNet-18 ˆπθ is then trained to minimize
the loss DKLpπ˚||ˆπθq on the training set. We compare the generalization performance of regular
Adam to a version that uses the AND-mask. For each method we ran an automatic hyperparameter
optimization study using Tree-structured Parzen Estimation (Bergstra et al., 2013) of 1024 trials.
Despite the theoretical computational efﬁciency of computing the AND-
mask as presented in Section 2.3 (i.e., linear time and memory in the size of
the mini-batch, just like classic SGD), current deep learning frameworks like
PyTorch (Paszke et al., 2017) have optimized routines that sum gradients
across examples in a mini-batch before it is possible to efﬁciently compute
the AND-mask. We therefore test the AND-mask in a slightly different way.
In training, in each iteration we sample a batch of data from a randomly
chosen level out of the 64 available (and cycle through them all once per
epoch). We then apply the AND-mask ‘temporally’, only allowing gradients
that are consistent across time (and therefore across levels). See Algorithm
1 in appendix B.6 for a detailed description of this alternative formulation
of the AND-mask. The ﬁgure shows the minimum test loss for the 10 best
runs, supporting the hypothesis that the AND-mask helps identify invariant
mechanisms across different levels.

4 RELATED WORK

Generalization and covariate shift. The classic formulation of statistical learning theory (Vapnik)
concerns learning from independent and identically distributed samples. The case where the distribu-
tion of the covariates at test time differs from the one observed during training is termed covariate
shift (Sugiyama et al., 2007; Quionero-Candela et al., 2009; Sugiyama and Kawanabe, 2012). Stan-
dard solutions involve re-weighting of the training examples, but require the additional assumption of
overlapping supports for train and test distributions.
Causal models and invariances. As we mentioned in the Introduction, causality provides a strong
motivation for our work, based on the notion that statistical dependencies are epiphenomena of an
underlying causal model (Pearl, 2009; Peters et al., 2017). The causal description identiﬁes stable
elements – e.g. physical mechanisms – connecting causes and effects, which are expected to remain
invariant under interventions or changing external conditions (Haavelmo, 1943; Schölkopf et al.,
2012)). This motivates our notion of invariant mechanisms, and inspired related notions which have
been proposed for robust regression (Rojas-Carulla et al., 2018; Heinze-Deml et al., 2018; Arjovsky
et al., 2019; Hermann and Lampinen, 2020; Ahuja et al., 2020; Krueger et al., 2020). We discuss this
in more detail in appendix C.1.
Domain generalization. ILC can be used in a setting of domain generalization (Muandet et al., 2013),
but it is not limited to it: as demonstrated in the experiments in Section 3.2, the AND-mask can be
applied even if domain labels are not available. In contrast, by treating every example as a single
domain, methods relying on domain classiﬁers (like DANN Ganin et al. (2016) or Balaji et al. (2018))
would require as many output units as there are training examples (i.e. 50’000 for CIFAR-10).
Gradient agreement. Looking at gradient agreement to learn meaningful representations in neural
networks has been explored in (Du et al., 2018; Eshratifar et al., 2018; Fort et al., 2019; Zhang et al.,
2019b). These approaches mainly rely on a measure of cosine similarity between gradients, which

9See Figure 17 in appendix B.6 for a visualization of the game.
10To obtain a robust evaluation, we preferred to approach behavioral cloning instead of the full RL problem, as
it is a standard supervised learning task and has substantially fewer moving parts than most deep RL algorithms.

8

AdamAdam+AND-mask1.651.701.751.80KL-divergence on test1e2Behavioral Cloningon CoinRunPublished as a conference paper at ICLR 2021

we did not consider here for two main reasons: (i) It is a ‘global’ property of the gradients, and it
would not allow us to extract precise information about different patterns in the network; (ii) It is
unclear how to extend it beyond pairs of vectors, and for pairwise interactions its computational cost
scales quadratic in the number of examples used.

5 CONCLUSIONS

Generalizing out of distribution is one of the most signiﬁcant open challenges in machine learning,
and relying on invariances across environments or examples may be key in certain contexts. In
this paper we analyzed how neural networks trained by averaging gradients across examples might
converge to solutions that ignore the invariances, especially if these are harder to learn than spurious
patterns. We argued that if learning signals are collected on one example at the time — as it is the
case for gradients, e.g., computed with backpropagation — the way these signals are aggregated
can play a signiﬁcant role in the patterns that will ultimately be expressed: Averaging gradients in
particular can be too permissive, acting as a logical OR of a collection of distinct patterns, and lead to
a ‘patchwork’ solution. We introduced and formalized the concept of Invariant Learning Consistency,
and showed how to learn invariances even in the face of alternative explanations that — although
spurious — fulﬁll most characteristics of a good solution. The AND-mask is but one of multiple
possible ways to improve consistency, and it is unlikely to be a practical algorithm for all applications.
However, we believe this should not distract from the general idea which we are trying to put forward
— namely, that it is worthwhile to study learning of explanations that are hard to vary, with the longer
term goal of advancing our understanding of learning, memorization and generalization.

ACKNOWLEDGMENTS

We wish to thank Sebastian Gomez, Luca Biggio, Julius von Kügelgen, Paolo Penna, Ioannis
Anagno, Ricards Marcinkevics, Sidak Pal Singh, Damien Teney for feedback on the manuscript, and
thank Nando de Freitas for fruitful discussions in the early stage of this project. We also thank the
Max Planck ETH Center for Learning Systems for supporting Giambattista Parascandolo, and the
International Max Planck Research School for Intelligent Systems for supporting Alexander Neitz.

REFERENCES

L. Adolphs, J. Kohler, and A. Lucchi. Ellipsoidal trust region methods and the marginal value of
hessian information for neural network training. arXiv preprint arXiv:1905.09201 (version 1),
2019.

K. Ahuja, K. Shanmugam, K. R. Varshney, and A. Dhurandhar. Invariant risk minimization games.
In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July
2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 145–155.
PMLR, 2020. URL http://proceedings.mlr.press/v119/ahuja20a.html.

T. Ando, C.-K. Li, and R. Mathias. Geometric means. Linear algebra and its applications, 385:

305–334, 2004.

M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz. Invariant risk minimization. arXiv preprint

arXiv:1907.02893, 2019.

Y. Balaji, S. Sankaranarayanan, and R. Chellappa. Metareg: Towards domain generalization using
meta-regularization. In S. Bengio, H. M. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
and R. Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference
on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal,
Canada, pages 1006–1016, 2018. URL https://proceedings.neurips.cc/paper/
2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html.

S. Becker, Y. Le Cun, et al. Improving the convergence of back-propagation learning with second
order methods. In Proceedings of the 1988 connectionist models summer school, pages 29–37,
1988.

9

Published as a conference paper at ICLR 2021

J. Bergstra, D. Yamins, and D. D. Cox. Making a science of model search: Hyperparameter
In Proceedings of the 30th
optimization in hundreds of dimensions for vision architectures.
International Conference on Machine Learning, ICML 2013, Atlanta, GA, USA, 16-21 June 2013,
volume 28 of JMLR Workshop and Conference Proceedings, pages 115–123. JMLR.org, 2013.
URL http://proceedings.mlr.press/v28/bergstra13.html.

L. Bottou, F. E. Curtis, and J. Nocedal. Optimization methods for large-scale machine learning. Siam

Review, 60(2):223–311, 2018.

K. Cobbe, O. Klimov, C. Hesse, T. Kim, and J. Schulman. Quantifying generalization in reinforcement
learning. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International
Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA,
volume 97 of Proceedings of Machine Learning Research, pages 1282–1289. PMLR, 2019a. URL
http://proceedings.mlr.press/v97/cobbe19a.html.

K. Cobbe, O. Klimov, C. Hesse, T. Kim, and J. Schulman. Quantifying generalization in reinforcement
learning. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International
Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA,
volume 97 of Proceedings of Machine Learning Research, pages 1282–1289. PMLR, 2019b. URL
http://proceedings.mlr.press/v97/cobbe19a.html.

K. Cobbe, C. Hesse, J. Hilton, and J. Schulman. Leveraging procedural generation to benchmark
reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning,
ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Re-
search, pages 2048–2056. PMLR, 2020. URL http://proceedings.mlr.press/v119/
cobbe20a.html.

G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control,

signals and systems, 2(4):303–314, 1989.

D. Deutsch. The beginning of inﬁnity: Explanations that transform the world. Penguin UK, 2011.

Y. Du, W. M. Czarnecki, S. M. Jayakumar, R. Pascanu, and B. Lakshminarayanan. Adapting auxiliary

losses using gradient similarity. arXiv preprint arXiv:1812.02224, 2018.

A. E. Eshratifar, D. Eigen, and M. Pedram. Gradient agreement as an optimization objective for

meta-learning. arXiv preprint arXiv:1810.08178, 2018.

S. Fort, P. K. Nowak, and S. Narayanan. Stiffness: A new perspective on generalization in neural

networks. arXiv preprint arXiv:1901.09491, 2019.

Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and
V. Lempitsky. Domain-adversarial training of neural networks. The Journal of Machine Learning
Research, 17(1):2096–2030, 2016.

T. Haavelmo. The statistical implications of a system of simultaneous equations. Econometrica, 11

(1), 1943.

C. Heinze-Deml and N. Meinshausen. Conditional variance penalties and domain shift robustness.

arXiv preprint arXiv:1710.11469, 2017.

C. Heinze-Deml, J. Peters, and N. Meinshausen. Invariant causal prediction for nonlinear models.

Journal of Causal Inference, 6(2), 2018.

K. L. Hermann and A. K. Lampinen. What shapes feature representations?

exploring
datasets, architectures, and training.
In H. Larochelle, M. Ranzato, R. Hadsell, M. Bal-
can, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual
Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12,
2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/
71e9c6620d381d60196ebe694840aaaa-Abstract.html.

K. D. Hoover. The logic of causal inference: Econometrics and the conditional analysis of causation.

Economics & Philosophy, 6(2):207–234, 1990.

10

Published as a conference paper at ICLR 2021

L. Hurwicz. On the structural form of interdependent systems. In E. Nagel, P. Suppes, and A. Tarski,
editors, Logic, Methodology and Philosophy of Science, Proceedings of the 1960 International
Congress, pages 232–239. Stanford University Press, Stanford, CA, 1962.

D. Janzing.

Causal

regularization.

In H. M. Wallach, H. Larochelle, A. Beygelz-
imer, F. d’Alché-Buc, E. B. Fox, and R. Garnett, editors, Advances in Neural Infor-
mation Processing Systems 32: Annual Conference on Neural Information Processing
Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages
12683–12693, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/
2172fde49301047270b2897085e4319d-Abstract.html.

S. Jastrz˛ebski, Z. Kenton, D. Arpit, N. Ballas, A. Fischer, Y. Bengio, and A. Storkey. Three factors

inﬂuencing minima in sgd. arXiv preprint arXiv:1711.04623, 2017.

D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In Y. Bengio and Y. LeCun,
editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA,
USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/
1412.6980.

D. Krueger, E. Caballero, J.-H. Jacobsen, A. Zhang, J. Binas, R. L. Priol, and A. Courville. Out-of-
distribution generalization via risk extrapolation (rex). arXiv preprint arXiv:2003.00688, 2020.

J. D. Lee, M. Simchowitz, M. I. Jordan, and B. Recht. Gradient descent converges to minimizers.

arXiv preprint arXiv:1602.04915, 2016.

S. Mandt, M. D. Hoffman, and D. M. Blei. Stochastic gradient descent as approximate bayesian

inference. The Journal of Machine Learning Research, 18(1):4873–4907, 2017.

J. M. Mooij, D. Janzing, J. Peters, and B. Schölkopf. Regression by dependence minimization and its
application to causal inference in additive noise models. In A. P. Danyluk, L. Bottou, and M. L.
Littman, editors, Proceedings of the 26th Annual International Conference on Machine Learning,
ICML 2009, Montreal, Quebec, Canada, June 14-18, 2009, volume 382 of ACM International
Conference Proceeding Series, pages 745–752. ACM, 2009. doi: 10.1145/1553374.1553470. URL
https://doi.org/10.1145/1553374.1553470.

K. Muandet, D. Balduzzi, and B. Schölkopf. Domain generalization via invariant feature repre-
sentation. In Proceedings of the 30th International Conference on Machine Learning, ICML
2013, Atlanta, GA, USA, 16-21 June 2013, volume 28 of JMLR Workshop and Conference Pro-
ceedings, pages 10–18. JMLR.org, 2013. URL http://proceedings.mlr.press/v28/
muandet13.html.

J. Nocedal and S. Wright. Numerical optimization. Springer Science & Business Media, 2006.

G. Parascandolo, N. Kilbertus, M. Rojas-Carulla, and B. Schölkopf. Learning independent causal
mechanisms. In J. G. Dy and A. Krause, editors, Proceedings of the 35th International Conference
on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018,
volume 80 of Proceedings of Machine Learning Research, pages 4033–4041. PMLR, 2018. URL
http://proceedings.mlr.press/v80/parascandolo18a.html.

A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga,

and A. Lerer. Automatic differentiation in pytorch, 2017.

J. Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2nd edition,

2009.

J. Peters, P. Bühlmann, and N. Meinshausen. Causal inference by using invariant prediction: iden-
tiﬁcation and conﬁdence intervals. Journal of the Royal Statistical Society: Series B (Statistical
Methodology), 78(5):947–1012, 2016.

J. Peters, D. Janzing, and B. Schölkopf. Elements of Causal Inference - Foundations and Learning

Algorithms. MIT Press, Cambridge, MA, USA, 2017.

J. Quionero-Candela, M. Sugiyama, A. Schwaighofer, and N. D. Lawrence. Dataset shift in machine

learning. The MIT Press, 2009.

11

Published as a conference paper at ICLR 2021

C. E. Rasmussen. Gaussian processes in machine learning. In Summer School on Machine Learning,

pages 63–71. Springer, 2003.

M. Rojas-Carulla, B. Schölkopf, R. Turner, and J. Peters. Invariant models for causal transfer learning.

The Journal of Machine Learning Research, 19(1):1309–1342, 2018.

B. Schölkopf. Causality for machine learning, 2019. arXiv:1911.10500.

B. Schölkopf, D. Janzing, J. Peters, E. Sgouritsa, K. Zhang, and J. M. Mooij. On causal and
anticausal learning. In Proceedings of the 29th International Conference on Machine Learning,
ICML 2012, Edinburgh, Scotland, UK, June 26 - July 1, 2012. icml.cc / Omnipress, 2012. URL
http://icml.cc/2012/papers/625.pdf.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization

algorithms. arXiv preprint arXiv:1707.06347, 2017.

H. A. Simon. Causal ordering and identiﬁability. In W. C. Hood and T. C. Koopmans, editors,
Studies in Econometric Methods, pages 49–74. John Wiley & Sons, New York, NY, 1953. Cowles
Commission for Research in Economics, Monograph No. 14.

S. P. Singh and D. Alistarh. Woodﬁsher: Efﬁcient second-order approximations for model compres-

sion. arXiv preprint arXiv:2004.14340, 2020.

A. Subbaswamy, P. Schulam, and S. Saria. Preventing failures due to dataset shift: Learning predictive
In K. Chaudhuri and M. Sugiyama, editors, The 22nd International
models that transport.
Conference on Artiﬁcial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha,
Okinawa, Japan, volume 89 of Proceedings of Machine Learning Research, pages 3118–3127.
PMLR, 2019. URL http://proceedings.mlr.press/v89/subbaswamy19a.html.

M. Sugiyama and M. Kawanabe. Machine learning in non-stationary environments: Introduction to

covariate shift adaptation. MIT press, 2012.

M. Sugiyama, M. Krauledat, and K.-R. Müller. Covariate shift adaptation by importance weighted

cross validation. Journal of Machine Learning Research, 8(May):985–1005, 2007.

V. N. Vapnik. The nature of statistical learning theory. Springer-Verlag New York, Inc. ISBN

0-387-94559-8.

J. von Kügelgen, A. Mey, and M. Loog. Semi-generative modelling: Covariate-shift adaptation with
cause and effect features. In K. Chaudhuri and M. Sugiyama, editors, The 22nd International
Conference on Artiﬁcial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha,
Okinawa, Japan, volume 89 of Proceedings of Machine Learning Research, pages 1361–1369.
PMLR, 2019. URL http://proceedings.mlr.press/v89/kugelgen19a.html.

C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning requires
rethinking generalization. In 5th International Conference on Learning Representations, ICLR
2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017.
URL https://openreview.net/forum?id=Sy8gdB9xx.

G. Zhang, L. Li, Z. Nado, J. Martens, S. Sachdeva, G. E. Dahl, C. J. Shallue, and R. B. Grosse.
Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model. In
H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. B. Fox, and R. Garnett, editors,
Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information
Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages
8194–8205, 2019a. URL https://proceedings.neurips.cc/paper/2019/hash/
e0eacd983971634327ae1819ea8b6214-Abstract.html.

Y. Zhang, W. Yu, and G. Turk. Learning novel policies for tasks. In K. Chaudhuri and R. Salakhutdi-
nov, editors, Proceedings of the 36th International Conference on Machine Learning, ICML 2019,
9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning
Research, pages 7483–7492. PMLR, 2019b. URL http://proceedings.mlr.press/
v97/zhang19q.html.

12

Published as a conference paper at ICLR 2021

A APPENDIX TO SECTION 2

A.1 A CLASSIC EXAMPLE OF A PATCHWORK SOLUTION

Consider a neural network with one hidden layer consisting of two neurons and sigmoidal activations:

fθpxq “ θ5σpθ1x ` θ2q ` θ6σpθ3x ` θ4q,

σpzq :“ 1{p1 ` e´zq.

(3)

We want to learn the continuous function f ˚ : r0, 1s Ñ r0, 2s deﬁned as

f ˚pxq “

$
x P r0, 0.4q;
0
x P r0.4, 0.5q;
10px ´ 0.4q
1
x P r0.5, 0.7q;
10px ´ 0.7q ` 1 x P r0.7, 0.8q;
2

’’’’’&
’’’’’%

x P r0.8, 1s.

To perform this task, we have access to (noiseless) data from two environments:

A : tpx, f pxqq | x P r0, 0.5qu, B : tpx, f pxqq | x P r0.5, 1su.

There is a simple constructive way, provided by the universal function approximation theorem Cy-
benko (1989) to ﬁt this function11 using fθ up to an arbitrarily small mean squared error LA`Bpθ˚q.
Leaving out the details of such a construction (Cybenko (1989) for details), the reader can check
on the left panel of Figure 10 that θ˚ “ p100, ´50, 100, ´75, 1, 1q provides a good ﬁt for both
environments A and B — both LApθ˚q and LBpθ˚q are small.

θ “ θ˚ “ p100, ´45, 100, ´75, 1, 1q

θ “ ˜θ˚ “ p100, ´45, 100, ´75, 1, ´0.5q

Figure 10: Performance of the neural network in Equation 3 for two different parameters. Any reasonable
modiﬁcation on θ6 (say ˘1) leaves the performance on environment A unchanged, while the performance on
environment B quickly degrades.

However, it is easy to realize that θ˚ — while being a solution which can be returned by gradient
descent using the pooled data A+B — is not consistent (formal deﬁnition given in the main paper in
Section 2). Indeed, it is possible to modify ˜θ˚ such that the loss in environment A remains almost
unchanged, while the loss in environment B gets larger. In particular, on the right panel of Figure 10,
we show that ˜θ˚ “ p100, ´50, 100, ´75, 1, ´0.5q is such that LApθ˚q ď LAp˜θ˚q ` (cid:15) (with (cid:15) very
small) but LBpθ˚q ! LBp˜θ˚q. According to our deﬁnition in Equation 1 (see main paper), we have
I (cid:15)pθ˚q ď |LBpθ˚q ´ LBp˜θ˚q| — that is a large number (low consistency).
Remark 1 (Connection to out of distribution generalization). The main point of this analysis was to
show an example of where our measure of consistency behaves according to expectations: A typical
implementation of the universal approximation theorem — which one would not expect to generalize
out of distribution, due to its ‘patchwork’ behavior — leads indeed to a very low consistency score.

11For a graphical description, the reader can check http://neuralnetworksanddeeplearning.

com/chap4.html

13

00.10.20.30.40.50.60.70.80.9100.20.40.60.811.21.41.61.8200.10.20.30.40.50.60.70.80.9100.20.40.60.811.21.41.61.82Published as a conference paper at ICLR 2021

A.2 SECTION 2.2: CONSISTENCY AS ARITHMETIC/GEOMETRIC MEAN OF LANDSCAPES

m

i“1 logpA´1

Geometric mean of matrices. Given an n-tuple of d ˆ d positive deﬁnite matrices pAjqn
j“1, the
geometric (Karcher) mean Ando et al. (2004) is the unique positive deﬁnite solution X to the equation
ř
i Xq “ 0, where log is the matrix logarithm. This matrix average has many desirable
properties, which make it relevant to signal processing and medical imaging. The Karcher mean
ř
m
i“1 dpAi, Xq2, where d is the Riemannian
can also be written as arg minXP
S
distance in the manifold of SPD matrices S ``pdq.

``pdq f pXq “ 1
2m

Figure 11: While the arithmetic mean of the two loss surfaces on the left is identical in all three cases (third
column), the geometric mean has weaker and weaker gradients (black arrow) the more inconsistent the two loss
surfaces become.

Link between consistency and geometric means. Here we show how the consistency score
introduced in Equation 1 can be linked (in a simpliﬁed setting) to a comparison between the arithmetic
and geometric means of the Hessians approximating the landscapes of two separate environments A
and B.
At the local minimizer θ˚ “ 0, we assume that LA “ LB “ 0 and consider the local quadratic
approximations LApθq “ 1
2 θJHBθ. Here, we make the additional
simplifying assumption that HA and HB are diagonal (or, more broadly, co-diagonalizable):
HA “ diagpλA
i ě 0 for all i “ 1, . . . , n.
The arithmetic and geometric means (noted as HA`B and HA^B) of these matrices are deﬁned in
this simpliﬁed setting as follows:
ˆ

2 θJHAθ and LBpθq “ 1

n q, HB “ diagpλB

i ě 0 and λB

n q, with λA

1 , ¨ ¨ ¨ , λB

1 , ¨ ¨ ¨ , λA

˙

˙

ˆb

b

HA`B “ diag

pλA

1 ` λB

1 q, ¨ ¨ ¨ ,

pλA

n ` λB
n q

, HA^B “ diag

1 λB
λA

1 , ¨ ¨ ¨ ,

n λB
λA
n

.

1
2

1
2

As motivated in the main paper and in Figure 12, one can link the consistency of two landscapes to a
comparison between the geometric and arithmetic means of the corresponding Hessians.
Proposition 3. In the setting we just described, the consistency score in Equation 1 can be estimated
as follows:

I (cid:15)pθ˚q ď 2(cid:15)

ˆ

˙

2

.

detpHA`Bq
detpHA^Bq

14

101101Loss env A101101Loss env B101101Arithmetic mean101101Geometric mean0.00.51.01.52.02.63.13.64.14.61011011011011011011011010.00.51.01.52.02.63.13.64.14.61011011011011011011011010.00.51.01.52.02.63.13.64.14.61011011011011011011011010.00.51.01.52.02.53.13.64.14.6Published as a conference paper at ICLR 2021

a

Figure 12: Plotted are contour lines θJH ´1θ “ 1 for HA “ diagp0.01, 1q and HB “ diagp1, 0.01q. It is
convenient to provide this visualization because it is linked to the matrix determinant: VolptθJH ´1θ “ 1uq “
π
detpHq. The geometric average retains the volume of the original ellipses, while the volume of HA`B is 25
times bigger. This magniﬁcation indicates that landscape A is not consistent with landscape B.

Before showing the proof, we note that the proposition gives a lower bound on the consistency.
That is, it provides a pessimistic estimate. Yet, as we motivated, this estimate has a nice geometric
interpretation. However, as we outline in a remark after the proof, this estimate is tight in two
important limit cases.

Proof. In this setting, Equation 1 gives

I (cid:15)pθ˚q :“ max

"

max
LApθqď(cid:15)

LBpθq, max

*

LApθq

.

Recall that

LApθq “

1
2

θJHAθ “

1
2

Hence, this is a simple quadratic program with quadratic constraints, and

LB pθqď(cid:15)
ÿ

i θ2
λA
i .

i

ÿ

1
2

i θ2
λB
i .

i

max
LApθqď(cid:15)

LBpθq “ max
i λA
i θ2
Further, we can change variables and introduce ˜θi “ θi
λB
i
λA
i

LBpθq “ max
}˜θ}2ď(cid:15)

max
LApθqď(cid:15)

ÿ

ř

1
2

i

i ď(cid:15)
a

All in all, we get

"

λA
i {2. The problem gets even simpler:

˜θ2
i “ (cid:15) ¨ max

i

λB
i
λA
i

.

I (cid:15)pθ˚q “ (cid:15) max

“ (cid:15) ¨ max

i

*

λA
i
λB
i
*

max
i

max
ˆ

,

, max
i
λA
i
λB
i
˙

λB
i
λA
i
"
λB
i
λA
i
λA
i
λB
i
i q2 ` pλA
i λA
λB
i
i ` λA
i q2
i λA
λB
i

`

λB
i
λA
i
pλB

pλB

*

i q2

*

.

ś

ď (cid:15) ¨ max

i

“ (cid:15) ¨ max

i

ď (cid:15) ¨ max

i

"

"

This means

a

I (cid:15)pθ˚q ď (cid:15) max

λB
i ` λA
ia
i λA
λB
i

pλB
i ` λA
a
i λA
λB
i
where the ﬁrst inequality comes from the monotonicity of the square root function, and the second
inequality comes from the fact that (i) the geometric mean is always smaller or equal than the
arithmetic mean and (ii) for any sequence of numbers αi ą 1, maxi αi ď

ipλB
i ` λA
a
ś
i λA
λB
i

detpHA`Bq
detpHA^Bq

“ 2(cid:15) max

i q{2

i q{2

ď 2(cid:15)

“ 2(cid:15)

ś

,

i

i

i

i αi.

15

-1-0.500.51-1-0.500.51-1-0.500.51-1-0.500.51Published as a conference paper at ICLR 2021

Remark 2 (Sanity check). There are two important cases where we can test the bound above. First, if
HA “ HB, then I (cid:15)pθ˚q “ (cid:15), and the bound returns I (cid:15)pθ˚q ď 2(cid:15), since the geometric and arithmetic
mean are the same. Next, say λA
i ą 0; then, both the bound and the inconsistency score
are 8 (highest possible inconsistency).

i “ 0 but λB

A.3 PROOF OF PROPOSITION 1

In this appendix section we consider the AND-masked GD algorithm, introduced at the end of
Section 2. We recall that the masked gradients at iteration k are mtpθkq d ∇Lpθkq, where mtpθkq
vanishes for any component where there are less than t P td{2 ` 1, . . . , du agreeing gradient signs
across environments, and is equal to one otherwise. In a full-batch setting, the algorithm is

θk`1 “ θk ´ η mtpθkq d ∇Lpθkq,

(AND-masked GD)

where η ą 0 is the learning rate.
Proposition 1. Let L have L-Lipschitz gradients and consider a learning rate η ď 1{L. After k
iterations, AND-masked GD visits at least once a point θ where }mtpθq d ∇Lpθq}2 ď Op1{kq.

Proof. Thanks to the component-wise L-smoothness and using a Taylor expansion around θi we
have

Lpθi`1q ď Lpθiq ´ ηx∇Lpθiq, mtpθiq d ∇Lpθiqy `

“ Lpθiq ´

η ´

}mtpθiq d ∇Lpθiq}2.

ˆ

˙

Lη2
2

Lη2
2

}mtpθiq d ∇Lpθiq}2

If we seek η ´ Lη2{2 ě η{2, then η ď 1
L , as we assumed in the proposition statement. Therefore,
Lpθi`1q ď Lpθiq ´ pη{2q}mtpθiq d ∇Lpθiq}2, for all i ě 0. Summing over i from 0 to a desired
iteration k, we get

k´1ÿ

pη{2q}mtpθiq d ∇Lpθiq}2 ď Lpθ0q ´ Lpθkq ď Lpθ0q.

i“0

Therefore,

min
i“0,...,k

}mtpθiq d ∇Lpθiq}2 ď

1
k

k´1ÿ

pη{2q}mtpθiq d ∇Lpθiq}2 ď

i“0

2Lpθ0q
ηk

.

Hence, there exist an iteration i˚ P t0, . . . , ku such that }mtpθi˚

q d ∇Lpθi˚

q}2 ď Op1{kq.

A.4 PROOF OF PROPOSITION 2

Here we ﬁx parameters θ P Rn and assume gradients ∇Lepθq P Rn coming from environments
e P E are drawn independently from a multivariate Gaussian with zero mean and σ2I covariance. We
want to show that, in this random setting, the AND-mask introduced in Section 2.3 decreases the
magnitude of the gradient step.

e“1 Le. While E}∇Lpθq}2 “
Proposition 2. Consider the setting we just outlined, with L “ p1{dq
Opn{dq, we have that @t P td{2 ` 1, . . . , du, Dc P p1, 2s such that E}mtpθq d ∇Lpθq}2 ď Opn{cdq.

ř

d

Proof. Let us drop the argument θ for ease of notation. First, let us consider ∇L (no gradient
AND-mask):

›
›
›
›
›

E

1
d

dÿ

i“1

∇Lei

“

1
d2

dÿ

i“1

E}∇Lei}2 “

nσ2
d

,

›
›
2
›
›
›

where in the ﬁrst equality we used the fact that the ∇Lei are uncorrelated and in the second the fact
that Er}∇Lei}2s is the trace of the covariance of ∇Lei.

16

Published as a conference paper at ICLR 2021

Next, assume we apply the element-wise AND-mask mt to the gradients, which puts to zero the
components (dimensions) where there are less than t P td{2, . . . , du equal signs. Since Gaussians are
symmetric around zero, the probability of having exactly u positive j-th gradient component among
d
d environments is P rppj “ uq “
. Hence, the probability to keep the j-th gradient direction
u
(considering also negative consistency) is

1
2

`

˘

`

˘

d

dÿ

d´tÿ

Prrrmtsj “ 1s “

Prppj “ uq `

Prppj “ uq

u“0
ˆ

1
2

˙

d d´tÿ

ˆ

k“0

˙

d
k

(4)

“

u“t
ˆ

1
2
ˆ

“ 2

˙

d dÿ

ˆ

˙

˙

k“t
d dÿ

k“t

1
2

d
k
ˆ

`

˙

.

d
k

¯›
›
2
›

´

›
›
›mt d

ř

1
d

d
i“1 ∇Lei

We would now like to compute E
. The difﬁculty lies in the fact that the
event mt “ 1 makes gradients conditionally dependent. Indeed, conditioning on both mt “ 1 and
r∇Lesj ą 0 changes the distribution of r∇Le1sj: this gradient entry is going to be more likely to be
positive or negative, depending on the value of r∇Lesj and on the details of the gradient mask. To
solve the issue, we our strategy is to reduce the discussion (without loss in generality and with no
additional assumption) to the case where gradient entries have all the same sign and hence conditional
independence is restored.

We consider the following writing for the quantity we are interested in:

˜

›
›
›
›
›mt d

E

1
d

dÿ

i“1

∇Lei

¸›
›
2
›
›
›

“

“

“

˜

1
d

»

nÿ

j“1

E

–rmtsj
»

nÿ

dÿ

E

j“1

ˆpj “0

–rmtsj
»

˜

nÿ

pd´tqÿ

dÿ

–

E

j“1

ˆpj “0

nÿ

dÿ

“ 2

ˆpj “t
»

˜

–

E

j“1

ˆpj “t

1
d

ﬁ

¸
2

r∇Leisj

ﬂ

dÿ

r∇Leisj

i“1

dÿ

r∇Leisj

i“1

dÿ

i“1
˜

1
d

1
d

dÿ

r∇Leisj

i“1

ﬁ

¸

2 ˇ
ˇ
ˇ
ˇpj “ ˆpj

¸

2 ˇ
ˇ
ˇ
ˇpj “ ˆpj

ﬂ Prrpj “ ˆpjs
ﬁ

ﬂ Prrpj “ ˆpjs

¸

2 ˇ
ˇ
ˇ
ˇpj “ ˆpj

ˆ

ﬁ

ﬂ

˙

d

ˆ

˙

,

d
ˆpj

1
2

where we used the deﬁnition of 2-norm, the law of total expectation, and the symmetry of the problem
with respect to positive and negative numbers. Finally, since the gradient components within the
same environment are conditionally independent, for any j P t1, . . . , nu we can write
˙

˜

¸

˜

ﬁ

»

ˆ

˙

ˆ

›
›
›
›
›mt d

E

1
d

dÿ

∇Lei

i“1

¸›
›
2
›
›
›

dÿ

–

E

“ 2n

ˆpj “t

dÿ

1
d

r∇Leisj

i“1

2 ˇ
ˇ
ˇ
ˇpj “ ˆpj

ﬂ

d

1
2

d
ˆpj

.

Finally, we note that the following bound holds:

»

–

E

¸

˜

1
d

dÿ

r∇Leisj

i“1

ﬁ

2 ˇ
ˇ
ˇ
ˇpj “ ˆpj ď d

ﬂ ď E

»

˜

–

1
d

dÿ

r∇Leisj

i“1

¸

2 ˇ
ˇ
ˇ
ˇpj “ d

ﬁ

ﬂ .

Indeed, if all environments lead to positive (or, symmetrically, negative) and non-interacting gradients
in the j-th direction, the average will be the biggest in norm. Moreover — crucially — conditioned
on the event pj “ d, gradients coming from different environments are distributed as a positive
half-normal distributions. Moreover, they are conditionally independent; this because, since they are

17

Published as a conference paper at ICLR 2021

all positive, the value of a gradient in one environment cannot inﬂuence the value of the gradient
in another one. We remark that conditional independence on the right-hand side is therefore not an
assumption, but is intrinsic to the upper bound.

Putting it all together, we have

˜

›
›
›
›
›mt d

E

1
d

dÿ

i“1

∇Lei

¸›
›
2
›
›
›

ď 2n

ď 2n

dÿ

ˆpj “t
dÿ

ˆpj “t

»

˜

–

E

σ2

r∇Leisj

1
d
˙

dÿ

i“1
ˆ

d

˙ ˆ

˙

d
ˆpj
˙

d´1

,

1
2

ˆ

1
2
ˆ

d
t

ď σ2npd ´ tq

¸

2 ˇ
ˇ
ˇ
ˇpj “ d

ˆ

ﬁ

ﬂ

˙
d

ˆ

˙

d
ˆpj

1
2

where in the second line we bounded the squared average of a sum of half normal distributions: let
tXiud
i“1 be a family of uncorrelated positive half-normal distributions derived from a Gaussians with
i s “ σ2. Also, ErXiXjs “
mean zero and variance σ2, we have12 that ErXis “ σ
ErXisErXjs ď σ2. Therefore,
»
˜

2{π and ErX 2

a

¸

ﬁ

–

E

1
d

dÿ

Xi

i“1

2

ﬂ “

1
d2

dÿ

i,j“1

ErXiXjs ď σ2.

Finally, if we set r “ t{d P p0.5, 1s, we have13
ˆ

ˆ

˙

d
t

„

˙

d

1
rrp1 ´ rq1´r
`

˘
d
t

as d Ñ 8 (discarding all polynomial terms). Hence
˘
`
1
quantity σ2npd ´ tq
2
t “ d{2, then we lose the exponential rate and get back to Opn{dq.

is of the form qd, with 1 ď q ă 2. So, the
d´1 will be exponentially decreasing at a rate Opn{p2 ´ qqdq. Notably, if

˘ `
d
t

12https://en.wikipedia.org/wiki/Half-normal_distribution
13Theorem 1 in Buri´c, Tomislav, and Neven Elezovi´c. “Asymptotic expansions of the binomial coefﬁcients.”

Journal of applied mathematics and computing 46.1-2 (2014): 135-145.

18

Published as a conference paper at ICLR 2021

B APPENDIX TO SECTION 3

We used Pytorch Paszke et al.
per.
learning-explanations-hard-to-vary.

in this pa-
Our codebase is publicly available at https://github.com/gibipara92/

to implement all experiments

(2017)

B.1 SECTION 3.1

Table 1: Hyperparameter ranges for synthetic data experiments. The regularizers L1 and L2 are never combined;
instead, one weight regularization type out of L1, L2 and none is selected and we sample from the respective
range afterwards.

Hyperparameter

No. hidden units
No. hidden layers
Batch-size
Optimizer
Learning rate
Batch-normalization
Dropout
L2 regularization
L1 regularization

Ranges

t256, 512u
t3, 5u
t64, 128, 256u
tAdamβ1“0.9,β2“0.999, SGD + momentum0.9u
t1e-3, 1e-2, 1e-1u
tYes, Nou
t0.0, 0.5u
t1e-5, 1e-4, 1e-3u
t1e-6, 1e-5, 1e-4u

B.2 DATASET

Here we report more technical details about the synthetic dataset described in
Section 3. Each example is constructed as follows: we ﬁrst choose the label
randomly to be either `1 or ´1, with equal probability. The example is a
vector with dS ` dM entries, consisting of the shortcut and the mechanism. In
our experiments, dM “ 2 and dS “ 32.

The Gaussian shortcuts are obtained by ﬁrst sampling one random vector
xs P RdS per environment. Its components xs,i are sampled independently
from a Normal distribution: xs,i „ N p0, 0.1q. We use xs for class 1, and ´xs
for class -1. In the test set, all shortcut components are sampled i.i.d. from the
same Normal distribution. Effectively, each example of the test set belongs to
a different domain. The mechanism is implemented as the two interconnected
spirals shown in Figure 13 by sampling the radius r „ Unifp0.08, 1.0q and
then computing the angle as α “ 2πnr where n is the number of revolutions of the spiral. We add
uniform noise in the range r´0.02, 0.02s to the radii afterwards.

Figure 13: The spirals
used as the mechanism
in the synthetic memo-
rization dataset.

The training dataset consists of 1280 examples per environment and we use D “ 32 environments
unless otherwise mentioned. The training datasets consists of 2000 examples.

B.3 EXPERIMENT

We train all networks for t3000{Du epochs, dropping the learning rate by a factor 10 halfway through,
and again at three-quarters of training. For computational reason, we stop each trial before completion
if the training accuracy exceeds 97% and the test accuracy is below 60%. All networks are MLPs with
LeakyReLU activation functions and a cross-entropy loss on the output. We run a hyperparameter
search over the ranges shown in Table 1. For IRM and the AND-mask, we select the best-performing
run and re-run it 50 times with different random seeds. For DANN and the standard baselines nothing
produced results signiﬁcantly better than chance.

B.3.1 STANDARD REGULARIZERS AND AND-MASK

The networks with the L1, L2, Dropout and Batch-normalization regularizers, have hyperparameters
that were randomly selected from Table 1. For the AND-mask we used the very same ranges. The

19

x1x2Published as a conference paper at ICLR 2021

regularizers L1 and L2 are never combined; instead, one weight regularization type out of L1, L2
and none is selected and we sample from the respective range afterwards. The parameters found to
work best from the grid search were: agreement threshold of 1, 256 hidden units, 3 hidden layers,
batch size 128, Adam with learning rate 1e-2, no batch norm, no dropout, L2-regularization with a
coefﬁcient of 1e-4, no L1-regularization. In practice, we often found it helpful to rescale the gradients
after masking to compensate for the decreasing overall magnitude. We add the option for gradient
rescaling as an additional hyperparameter, as we found it to help in several experiments. It rescales
gradient components layer-wise after masking, by multiplying the remaining gradient components
by c, where c is the ratio of the number of components in that layer over the number of non-masked
components in that layer (i.e. the sum of the binary elements in the mask).14. We speculate that for
very large layers, a less extreme normalization scheme or the additional use of gradient clipping
might be appropriate.

B.3.2 DOMAIN ADVERSARIAL NEURAL NETWORKS

The experiments using DANN follow a similar pattern. The model consists of an embedding network,
a classiﬁcation network, and a “domain discrimination” network. All three modules are two-layer
multi-layer perceptrons (MLP). The number of hidden units of all MLPs are sampled from the range
speciﬁed in Table 1, and we trained 100 models. Both label classiﬁer and domain discriminator
are applied to the output of the embedding network. The label classiﬁer is trained to minimize the
cross-entropy-loss between the predicted and the true label. Similarly, the domain discriminator is
trained to minimize the loss between predicted and true domain-label. The embedding network is
trained to minimize the regular task classiﬁcation loss and at the same time to maximize the the
domain-loss achieved by the domain discriminator.

B.3.3

INVARIANT RISK MINIMIZATION

For the experiments using IRM we used the authors’ PyTorch implementation from https:
//github.com/facebookresearch/InvariantRiskMinimization. We perform a
random hyperparameter search over with the ranges shown in Table 2

Table 2: Hyperparameter ranges for IRM.

Hyperparameter

Ranges

No. hidden units
No. hidden layers
Batch-size
Optimizer
Batch-normalization
Penalty weight
Number of annealing iterations
Learning rate

t256, 512u
t3, 5u
t64, 128, 256u
tAdamβ1“0.9,β2“0.999, SGD + momentum0.9u
tYes, Nou
t10.0, 100.0, 1000.0u
t0, 1, 2, 4, 8u
t1e-3, 1e-2, 1e-1, 1u

B.3.4 CURVES FOR ALL EXPERIMENTS

In Figure 14 we show the learning curves of training and test accuracy for the different methods.

B.3.5 CORRELATION PLOTS

For the correlation plots in Figure 7 we used a randomly initialized MLP with the following conﬁgu-
ration: 3 hidden layers, 256 hidden units. The dataset was using 16 environments and batches of size
1024. The lines in Figure 7 are linear least-squares regressions to the gradient data shown as scatter
plots. We repeat the experiment 10 times with different network weight seeds, resulting in the 10
regression lines. Zero gradients are excluded from the regression computation, as most gradients are
masked out by the product mask in both cases.

14Therefore, c is 1 if the AND-mask has only 1s, and inﬁnite if all components are masked out (which we

then keep as 0.)

20

Published as a conference paper at ICLR 2021

Figure 14: Learning curves for the evaluated methods. The top row shows the accuracy on the training set, the
bottom row shows the accuracy on the test set.

B.4 FURTHER VISUALIZATIONS AND EXPERIMENTS

In Figure 15 we show how many environments need to be present for the baseline without AND-mask
to switch the decision boundary from the shortcuts to the mechanism. Under the same experimental
condition as in the main paper, the baseline ﬁrst succeeds at 1024 environments.

Figure 15: Relationship between number of training environments and test accuracy for the AND-mask method
compared to the baseline. We show the best performance out of ﬁve runs using the settings that were used for
the experiment in the main text.

B.5 SECTION 3.2: CIFAR-10 MEMORIZATION AND LABEL NOISE EXPERIMENTS

Memorization experiment
In Figure 16, we report the test
performance (dashed lines) corresponding to the curves pre-
sented in the main paper for the CIFAR-10 memorization
experiment. The test performance with standard labels de-
creases slower than the training performance as the threshold
increases, and they eventually reach the same value. This is
consistent with the hypothesis that by training on the consis-
tent directions, the AND-mask selects the invariant patterns
and prunes out the signals that are not invariant.

Figure 16: Dashed lines show test acc,
solid lines show training acc.

Network architecture and training details Each trial
trains the ResNet “FastResNet” from the PyTorch-Ignite example15 for 80 epochs on the full
CIFAR-10 training set. We use the Adam optimizer with a learning rate of 5e´4, and a 0.1 learning
rate decay at epoch 40 and 60. We ﬁx the batch size to 80. We set up 14 trials by evaluating each of the
AND-mask-thresholds t0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8u for two datasets: (a) unchanged CIFAR-10, (b)
CIFAR-10 with the training labels replaced by random labels. Note that a threshold of 0 corresponds
to not using the AND-mask. Each trial is run twice with separate random seeds.

Label noise experiment We trained the same ResNet as for the experiment above, once with and
once without the AND-mask. We ran each experiment with three different starting learning rates

15https://github.com/pytorch/ignite/blob/master/examples/contrib/

cifar10/fastresnet.py

21

0.500.751.00Training accuracyBaselineDANNIRMAND-mask11050Epoch0.500.751.00Test accuracy11050Epoch11050Epoch11050Epoch248163225651210242048409664 1280.51.0Max. accuracyBaselineProd-maskTraining environments      0.00      0.05      0.10      0.20      0.40      0.60      0.80Agreement threshold0.250.500.751.00AccuraciesAND-maskStandard LabelsRandom LabelsPublished as a conference paper at ICLR 2021

t5e´4, 1e´3, 5e´3u and a learning rate decay at epoch 60. The baseline worked best with a learning
rate of 1e´3, while the AND-mask with 5e´3, likely to compensate for the masked out gradients.
The AND-mask threshold that worked best was 0.2, which is consistent with the results obtain in the
experiment above.

B.6 SECTION 3.3: BEHAVIORAL CLONING ON COINRUN

The target policy π˚ is obtained by training PPO (Schulman et al., 2017) for 400M time steps
using the code16 for the paper Cobbe et al. (2020). This policy is trained on the full distribution of
levels in order to maximize its generality. We use π˚ to generate a behavioral cloning (BC) dataset,
consisting of pairs ps, π˚pa|sqq, where s are the input-images (64 ˆ 64 RGB) and π˚pa|sq is the
discrete probability distribution over actions output by π˚.
The states are sampled randomly from trajectories generated by π˚. In order to test for generalization
performance, the BC training dataset is restricted to 64 distinct levels. We generate 1000 examples
per training level. The test set consists of 2000 examples, each from a different level which does not
appear in the training set.

Figure 17: Screenshots of 6 levels of CoinRun (from OpenAI).

A ResNet-18 ˆπθ is trained to minimize the loss DKLpπ˚||ˆπθq. We ran two automatic hyperparameter
optimization studies using Tree-structured Parzen Estimation (TPE) (Bergstra et al., 2013) of 1024
trials each, with and without the AND-mask. The learning rate was decayed by a factor of 10 half-way
at at 3{4 of the training epochs.

The “temporal” version of the AND-mask used for this experiment is reported in Algorithm 1.

Algorithm 1: Temporal AND-mask Adam

1 m Ð β1 ¨ m ` p1 ´ β1q ¨ g
2 v Ð β2 ¨ v ` p1 ´ β2q ¨ pg ˝ gq
3 a Ð β3 ¨ a ` p1 ´ β3q ¨ elemwise_signpgq
4 b Ð 1r|a| ě τ s
?
5 θ Ð θ ´ αpm ˝ bq m

v ` (cid:15)

In blue we highlight the additional lines compared to traditional Adam. The threshold τ and β3
are hyperparameters that we included in the 1’024 trials of the search using Tree-structured Parsen
Estimators. For the top 10 runs, hyperparameter values that were selected via the TPE search for the
AND-mask are the following.

Table 3: Hyperparameters for the 5 best runs using the AND-mask, from the TPE search.

Test KL div

1.652e-2
1.656e-2
1.662e-2
1.665e-2
1.672e-2

lr

0.0078
0.0072
0.0080
0.0068
0.0063

β1
0.21
0.26
0.23
0.33
0.67

β3
0.79
0.86
0.84
0.72
0.65

τ

0.36
0.40
0.41
0.47
0.47

weight decay

0.057
0.041
0.045
0.077
0.080

We found that applying weight decay as a second independent update after the AND-mask routine
improved performance. To keep the comparison fair, we added this as a switch in the hyperparameter
search for the Adam baseline as well, and it improved performance there as well.

16https://github.com/openai/train-procgen

22

Published as a conference paper at ICLR 2021

Figure 18: Learning curves for the behavioral cloning experiment on CoinRun. Training loss is shown on the
left, test loss is shown on the right. We show the mean over the top-10 runs for each method. The shaded regions
correspond to the 95% conﬁdence interval of the mean based on bootstrapping.

C APPENDIX TO SECTION 4

C.1 RELATED WORK IN CAUSAL INFERENCE

Causal graphs and causal factorizations The formalization of causality through directed acyclic
graphs (Pearl, 2009) is a key element informing our exposition. According to such formalization,
a causal model gives rise to each observed distribution. It is thereby possible to exploit properties
of the causal factorization of the joint probability distribution over the observed variables. Clearly,
there are many ways to factorize a joint distribution into conditionals; a distinguishing feature
of the causal factorization is that many of the conditionals, which we can think of as physical
mechanisms underlying the statistical dependencies represented, are expected to remain invariant
under interventions or changing external conditions. This postulate has appeared in various forms in
the literature (Haavelmo, 1943; Simon, 1953; Hurwicz, 1962; Pearl, 2009; Schölkopf et al., 2012).17

Causal models and robust regression Based on this insight, it was proposed that regression based
on causal features should presents desirable invariance and robustness properties (Mooij et al., 2009;
Schölkopf et al., 2012; Peters et al., 2016; Rojas-Carulla et al., 2018; Heinze-Deml et al., 2018; von
Kügelgen et al., 2019; Parascandolo et al., 2018). In this view, the mechanisms can be considered
as features of the patterns such that they support stable conditional probabilities. Thus learning the
mechanisms may help achieve a stable performance across a number of conditions. Other works
connecting causality and learning through invariances are (Subbaswamy et al., 2019; Heinze-Deml
and Meinshausen, 2017), and perhaps – most related to our work – (Arjovsky et al., 2019): we
presented a comparison with this method in the following section.

Causal regularization Recently (Janzing, 2019) showed that biasing learning towards models of
lower complexity might in some cases be beneﬁcial for a notion of generalization from observational
to interventional regimes. Our proposed solution is however different, in that we only indirectly deal
with penalizing model complexity, and rather focus on our proposed notion of consistency.

C.2 LEARNING INVARIANCES IN THE DATA

Here we are going to compare ILC to other approaches for learning invariances in the data with
neural networks, and in particular to Invariant Risk Minimization (IRM) Arjovsky et al. (2019).
The authors of IRM analyze a set up where minimizing training error might lead to models which
absorb all the correlations found within the training data, thus failing to recover the relevant causal
explanation. They consider a multi-environment setting and focus on the objective of extracting data
representations that lead to invariant prediction across environments.

While the high level objective is close to the one we focused on, the differences become clear when
considering the deﬁnition of invariant predictors presented in Arjovsky et al. (2019):

17This would be different for a non-causal factorization of the joint distribution, see Schölkopf (2019)

23

11530Epoch103102Train DKLAdamAND-mask11530Epoch2×1023×102Test DKLAdamAND-maskPublished as a conference paper at ICLR 2021

Deﬁnition 1. A data representation Φ : X Ñ H elicits an invariant predictor w ˝ Φ across envi-
ronments E if there is a classiﬁer w : H Ñ Y simultaneously optimal for all environments, i.e.,
w P arg min ¯w:

Ñy Rep ¯w ˝ Φq @e P E.

H

In particular, the objective minimized by IRM is:

ÿ

min
Ñ
X

Φ:

Y

eP

Etr

RepΦq ` λ ¨

›
›
›2
›∇w|w“1.0Repw ¨ Φq

(5)

where Φ are the logits predicted by the neural network and w is a dummy scaling variable (see
›
›
›2: One way to
›∇w|w“1.0Repw ¨ Φq
Arjovsky et al. (2019)). The relevant part is the penalty term λ ¨
interpret it, is that the penalty is large on every environment where the distribution outputted by Φ
could be made ‘closer’ to the distribution of the labels by either sharpening (w ą 1) or softening it
(i.e., closer to uniform w ă 1).

Let us consider the example from IRM, where the authors describe two datasets of images that each
contain either a cow or a camel: In one of the datasets, there is grass on 80% of the images with cows,
while in the other dataset there is grass on 90% of them. IRM then makes the point that we can learn
to ignore grass as a feature, because its correlation with the label cow is inconsistent (80% vs 90%).
The setting we consider in this paper is slightly different: take our example from the CIFAR-10
experiments. Under our concept of invariance, we expect that (depending on the data generating
process) even a single dataset where we treat every image as coming from its own ‘environment’
should be sufﬁcient to discover invariances. Drawing a connection to the setting from IRM, we would
argue that the second dataset should not be necessary to learn that ‘grass’ is not ‘cow’. If one treats
every example as coming from its own environment, there is already sufﬁcient information in the ﬁrst
dataset to realize that cows are not grass: Grass is predictive of cows only in 80% of the data, so grass
cannot be ‘cow’. The actual cow on the other hand, should be present in 100% of the images, and as
such it is the invariance we are looking for. Note that this is of course a much more strict deﬁnition of
invariance: If our dataset contains images labeled as ’cows’ but that have no cows within them, we
might start to discard the features of cows as well.

24

