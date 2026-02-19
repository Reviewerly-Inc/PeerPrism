Published as a conference paper at ICLR 2022

HYPERDQN: A RANDOMIZED EXPLORATION METHOD
FOR DEEP REINFORCEMENT LEARNING

Ziniu Li1, Yingru Li1† , Yushun Zhang1, Tong Zhang2†, and Zhi-Quan Luo1†
1Shenzhen Research Institute of Big Data,
The Chinese University of Hong Kong, Shenzhen, China
2Hong Kong University of Science and Technology
{ziniuli, yingruli, yushunzhang}@link.cuhk.edu.cn,
tongzhang@ust.hk, luozq@cuhk.edu.cn

ABSTRACT

Randomized least-square value iteration (RLSVI) is a provably efficient exploration
method. However, it is limited to the case where (1) a good feature is known in
advance and (2) this feature is fixed during the training. If otherwise, RLSVI
suffers an unbearable computational burden to obtain the posterior samples. In
this work, we present a practical algorithm named HyperDQN to address the
above issues under deep RL. In addition to a non-linear neural network (i.e., base
model) that predicts Q-values, our method employs a probabilistic hypermodel
(i.e., meta model), which outputs the parameter of the base model. When both
models are jointly optimized under a specifically designed objective, three purposes
can be achieved. First, the hypermodel can generate approximate posterior samples
regarding the parameter of the Q-value function. As a result, diverse Q-value
functions are sampled to select exploratory action sequences. This retains the
punchline of RLSVI for efficient exploration. Second, a good feature is learned to
approximate Q-value functions. This addresses limitation (1). Third, the posterior
samples of the Q-value function can be obtained in a more efficient way than
the existing methods, and the changing feature does not affect the efficiency.
This deals with limitation (2). On the Atari suite, HyperDQN with 20M frames
outperforms DQN with 200M frames in terms of the maximum human-normalized
score. For SuperMarioBros, HyperDQN outperforms several exploration bonus
and randomized exploration methods on 5 out of 9 games.

1

INTRODUCTION

Reinforcement learning (RL) (Sutton & Barto, 2018) involves an agent that interacts with an unknown
environment to maximize cumulative reward. The trade-off between exploration and exploitation is
a fundamental problem in RL (Kakade, 2003). On the one hand, the agent needs to explore highly
uncertain states and actions, which may sacrifice immediate reward. On the other hand, in the long
term, the agent should take the best-known action; however, this action may be sub-optimal due to
partial information. To this end, sample-efficient RL agents should qualify the epistemic uncertainty
(i.e., subjective uncertainty due to limited samples (Osband, 2016a)) to address the trade-off.

Bayesian approaches like Thompson sampling (Russo et al., 2018) provide a nice way of encoding the
epistemic uncertainty by posterior distribution. For instance, randomized least-square value iteration
(RLSVI)1 (Osband et al., 2016b; 2019) is a well-known Bayesian algorithm. Specifically, RLSVI
takes three steps to address the exploration and exploitation trade-off. First, conditioned on observed
data, RLSVI solves a Bayesian linear regression problem and updates the posterior distribution over
(the parameter of) the optimal Q-value functions by the Bayes update rule. Second, RLSVI samples a
specific Q-value function from the posterior distribution. Third, to collect more data, greedy actions
are selected based on this Q-value function. In theory, the randomness in posterior sampling could
yield positive bias, which boosts optimistic behaviors (Osband et al., 2019; Russo, 2019; Zanette
et al., 2020). However, RLSVI is restricted to the following case.

†: Corresponding authors.
1For readers who are not familiar with RLSVI, please refer to Appendix A.1 for a brief introduction.

1

Published as a conference paper at ICLR 2022

• A good feature is known in advance. Over this feature, the Q-values can be approximated as a

linear function.

• This feature is fixed so that the posterior distribution is easy to compute in an incremental update

way (see Remark 2 in Appendix A.1 or (1.1) for an illustration).

Unfortunately, these two requirements do not hold in the deep RL scenario. The challenges of
extending RLSVI to deep RL are listed below.

1) If the provided feature is not good and fixed, RLSVI is rarely competent. Under the deep
RL setting, only a raw feature outputted by a randomly initialized neural network is available.
Such an unpolished feature has limited representation power, so the Q-value function cannot be
approximated accurately. Further, the mechanism of RLSVI only involves updating the model
parameter, while the feature is left to be fixed (as a raw feature in this setting). As a result, the
empirical performance of RLSVI is often poor; see Appendix E.1 for the evidence.

2) When we update the feature over iterations in deep RL, the computational complexity of
RLSVI becomes unbearable. As the data accumulates over iterations, RLSVI needs to repeatedly
compute the feature covariance matrix to update the posterior distribution. With the fixed feature
mapping, this process can be efficiently implemented in an incremental way (see (1.1), where xK
denotes the state-action pair (sK, aK) at iteration K and ϕ : S × A → Rd is a feature mapping.).
However, when the feature mapping ϕK is changing over iterations, we need to repeatedly calculate
the matrix ΦK using all historical data as in (1.2) (e.g. in Atari, this calculation could involve more
than 1M samples with dimension 512), and this process results in a huge computation burden.

fixed ϕ: ΦK = ΦK−1 + ϕ(xK)ϕ(xK)⊤ with Φ0 = 0,

(1.1)

changing ϕK: ΦK :=

K
(cid:88)

ℓ=1

ϕK(xℓ)ϕK(xℓ)⊤, ΦK−1 :=

K−1
(cid:88)

ℓ=1

ϕK−1(xℓ)ϕK−1(xℓ)⊤, · · ·

(1.2)

To address these two issues in deep RL, Bootstrapped DQN (BootDQN) (Osband et al., 2016a; 2018)
takes a remarkable first step. To bypass the hurdle of issues 1 and 2, BootDQN simultaneously
trains tens of ensembles (i.e., randomly initialized neural networks) and views them as approximate
posterior samples of Q-value functions. However, with limited computation resources, BootDQN
typically uses ensembles with a strictly limited number of elements. As such, the quality of the
learned posterior distribution could be poor (Lu & Van Roy, 2017).

In this work, we present a principled approach named HyperDQN that addresses the above issues
under the context of deep RL. Specifically, HyperDQN is built on two parametric models: in addition
to a non-linear neural network (i.e., base model) that predicts Q-values, it employs a hypermodel
(Dwaracherla et al., 2020) (i.e., meta model). This hypermodel maps a vector z (from the Gaussian
distribution) to a parameter instance of the Q-value function. As shown in Figure 1, this hypermodel
aims to serve as a proxy for generating the posterior samples of θpredict. To achieve this goal, both
models are jointly optimized by a specially designed temporal difference (TD) objective (see (4.2)).

Figure 1: Illustration for HyperDQN .

After optimization, our approach has the following essential merits.
• First, we prove that the optimized hypermodel can approximate the posterior distribution under
the Bayesian linear regression case (see Theorem 1). Thus, diverse Q-value functions can be
sampled to select exploratory action sequences. This retains the punchline of RLSVI for efficient
exploration. We remark that this theoretical guarantee is missing in (Dwaracherla et al., 2020).

2

state<latexit sha1_base64="tkJlAyuos0Rl1U5c9sVcJ4d6hqs=">AAAC0XicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFl047KifUBbJUmndWiahMlELKUgbv0Bt/pT4h/oX3hnHEEtohOSnDn3njNz7/WTkKfScV5y1szs3PxCfrGwtLyyulZc36incSYCVgviMBZN30tZyCNWk1yGrJkI5g39kDX8wbGKN66ZSHkcnctRwjpDrx/xHg88SdRFW7IbKeU4lZ5kk8tiySk7etnTwDWgBLOqcfEZbXQRI0CGIRgiSMIhPKT0tODCQUJcB2PiBCGu4wwTFEibURajDI/YAX37tGsZNqK98ky1OqBTQnoFKW3skCamPEFYnWbreKadFfub91h7qruN6O8bryGxElfE/qX7zPyvTtUi0cOhroFTTYlmVHWBccl0V9TN7S9VSXJIiFO4S3FBONDKzz7bWpPq2lVvPR1/1ZmKVfvA5GZ4U7ekAbs/xzkN6ntl1ym7p/ulypEZdR5b2MYuzfMAFZygihp5CzzgEU/WmTWybq27j1QrZzSb+Las+3fyfZWp</latexit><latexit sha1_base64="tkJlAyuos0Rl1U5c9sVcJ4d6hqs=">AAAC0XicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFl047KifUBbJUmndWiahMlELKUgbv0Bt/pT4h/oX3hnHEEtohOSnDn3njNz7/WTkKfScV5y1szs3PxCfrGwtLyyulZc36incSYCVgviMBZN30tZyCNWk1yGrJkI5g39kDX8wbGKN66ZSHkcnctRwjpDrx/xHg88SdRFW7IbKeU4lZ5kk8tiySk7etnTwDWgBLOqcfEZbXQRI0CGIRgiSMIhPKT0tODCQUJcB2PiBCGu4wwTFEibURajDI/YAX37tGsZNqK98ky1OqBTQnoFKW3skCamPEFYnWbreKadFfub91h7qruN6O8bryGxElfE/qX7zPyvTtUi0cOhroFTTYlmVHWBccl0V9TN7S9VSXJIiFO4S3FBONDKzz7bWpPq2lVvPR1/1ZmKVfvA5GZ4U7ekAbs/xzkN6ntl1ym7p/ulypEZdR5b2MYuzfMAFZygihp5CzzgEU/WmTWybq27j1QrZzSb+Las+3fyfZWp</latexit><latexit sha1_base64="tkJlAyuos0Rl1U5c9sVcJ4d6hqs=">AAAC0XicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFl047KifUBbJUmndWiahMlELKUgbv0Bt/pT4h/oX3hnHEEtohOSnDn3njNz7/WTkKfScV5y1szs3PxCfrGwtLyyulZc36incSYCVgviMBZN30tZyCNWk1yGrJkI5g39kDX8wbGKN66ZSHkcnctRwjpDrx/xHg88SdRFW7IbKeU4lZ5kk8tiySk7etnTwDWgBLOqcfEZbXQRI0CGIRgiSMIhPKT0tODCQUJcB2PiBCGu4wwTFEibURajDI/YAX37tGsZNqK98ky1OqBTQnoFKW3skCamPEFYnWbreKadFfub91h7qruN6O8bryGxElfE/qX7zPyvTtUi0cOhroFTTYlmVHWBccl0V9TN7S9VSXJIiFO4S3FBONDKzz7bWpPq2lVvPR1/1ZmKVfvA5GZ4U7ekAbs/xzkN6ntl1ym7p/ulypEZdR5b2MYuzfMAFZygihp5CzzgEU/WmTWybq27j1QrZzSb+Las+3fyfZWp</latexit><latexit sha1_base64="tkJlAyuos0Rl1U5c9sVcJ4d6hqs=">AAAC0XicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFl047KifUBbJUmndWiahMlELKUgbv0Bt/pT4h/oX3hnHEEtohOSnDn3njNz7/WTkKfScV5y1szs3PxCfrGwtLyyulZc36incSYCVgviMBZN30tZyCNWk1yGrJkI5g39kDX8wbGKN66ZSHkcnctRwjpDrx/xHg88SdRFW7IbKeU4lZ5kk8tiySk7etnTwDWgBLOqcfEZbXQRI0CGIRgiSMIhPKT0tODCQUJcB2PiBCGu4wwTFEibURajDI/YAX37tGsZNqK98ky1OqBTQnoFKW3skCamPEFYnWbreKadFfub91h7qruN6O8bryGxElfE/qX7zPyvTtUi0cOhroFTTYlmVHWBccl0V9TN7S9VSXJIiFO4S3FBONDKzz7bWpPq2lVvPR1/1ZmKVfvA5GZ4U7ekAbs/xzkN6ntl1ym7p/ulypEZdR5b2MYuzfMAFZygihp5CzzgEU/WmTWybq27j1QrZzSb+Las+3fyfZWp</latexit>valuefunction<latexit sha1_base64="NiyaUwDTuG/J8yfcgRalNLoMb5s=">AAAC3HicjVHLSsNAFD2Nr1pfURcu3ASL4KokIuiy6MZlBfuAtpZkOq2heZFMiqV0507c+gNu9XvEP9C/8M6YglpEJyQ5c+49Z+6d60SemwjTfM1pc/MLi0v55cLK6tr6hr65VUvCNGa8ykIvjBuOnXDPDXhVuMLjjSjmtu94vO4MzmS8PuRx4obBpRhFvO3b/cDtucwWRHX0nZbgN0KI8dD2Um700oDJwKSjF82SqZYxC6wMFJGtSqi/oIUuQjCk8MERQBD2YCOhpwkLJiLi2hgTFxNyVZxjggJpU8rilGETO6Bvn3bNjA1oLz0TpWZ0ikdvTEoD+6QJKS8mLE8zVDxVzpL9zXusPGVtI/o7mZdPrMA1sX/pppn/1cleBHo4UT241FOkGNkdy1xSdSuycuNLV4IcIuIk7lI8JsyUcnrPhtIkqnd5t7aKv6lMyco9y3JTvMsqacDWz3HOgtphyTJL1sVRsXyajTqPXezhgOZ5jDLOUUFV1f+IJzxrV9qtdqfdf6ZquUyzjW9Le/gAcxGZtg==</latexit><latexit sha1_base64="NiyaUwDTuG/J8yfcgRalNLoMb5s=">AAAC3HicjVHLSsNAFD2Nr1pfURcu3ASL4KokIuiy6MZlBfuAtpZkOq2heZFMiqV0507c+gNu9XvEP9C/8M6YglpEJyQ5c+49Z+6d60SemwjTfM1pc/MLi0v55cLK6tr6hr65VUvCNGa8ykIvjBuOnXDPDXhVuMLjjSjmtu94vO4MzmS8PuRx4obBpRhFvO3b/cDtucwWRHX0nZbgN0KI8dD2Um700oDJwKSjF82SqZYxC6wMFJGtSqi/oIUuQjCk8MERQBD2YCOhpwkLJiLi2hgTFxNyVZxjggJpU8rilGETO6Bvn3bNjA1oLz0TpWZ0ikdvTEoD+6QJKS8mLE8zVDxVzpL9zXusPGVtI/o7mZdPrMA1sX/pppn/1cleBHo4UT241FOkGNkdy1xSdSuycuNLV4IcIuIk7lI8JsyUcnrPhtIkqnd5t7aKv6lMyco9y3JTvMsqacDWz3HOgtphyTJL1sVRsXyajTqPXezhgOZ5jDLOUUFV1f+IJzxrV9qtdqfdf6ZquUyzjW9Le/gAcxGZtg==</latexit><latexit sha1_base64="NiyaUwDTuG/J8yfcgRalNLoMb5s=">AAAC3HicjVHLSsNAFD2Nr1pfURcu3ASL4KokIuiy6MZlBfuAtpZkOq2heZFMiqV0507c+gNu9XvEP9C/8M6YglpEJyQ5c+49Z+6d60SemwjTfM1pc/MLi0v55cLK6tr6hr65VUvCNGa8ykIvjBuOnXDPDXhVuMLjjSjmtu94vO4MzmS8PuRx4obBpRhFvO3b/cDtucwWRHX0nZbgN0KI8dD2Um700oDJwKSjF82SqZYxC6wMFJGtSqi/oIUuQjCk8MERQBD2YCOhpwkLJiLi2hgTFxNyVZxjggJpU8rilGETO6Bvn3bNjA1oLz0TpWZ0ikdvTEoD+6QJKS8mLE8zVDxVzpL9zXusPGVtI/o7mZdPrMA1sX/pppn/1cleBHo4UT241FOkGNkdy1xSdSuycuNLV4IcIuIk7lI8JsyUcnrPhtIkqnd5t7aKv6lMyco9y3JTvMsqacDWz3HOgtphyTJL1sVRsXyajTqPXezhgOZ5jDLOUUFV1f+IJzxrV9qtdqfdf6ZquUyzjW9Le/gAcxGZtg==</latexit><latexit sha1_base64="NiyaUwDTuG/J8yfcgRalNLoMb5s=">AAAC3HicjVHLSsNAFD2Nr1pfURcu3ASL4KokIuiy6MZlBfuAtpZkOq2heZFMiqV0507c+gNu9XvEP9C/8M6YglpEJyQ5c+49Z+6d60SemwjTfM1pc/MLi0v55cLK6tr6hr65VUvCNGa8ykIvjBuOnXDPDXhVuMLjjSjmtu94vO4MzmS8PuRx4obBpRhFvO3b/cDtucwWRHX0nZbgN0KI8dD2Um700oDJwKSjF82SqZYxC6wMFJGtSqi/oIUuQjCk8MERQBD2YCOhpwkLJiLi2hgTFxNyVZxjggJpU8rilGETO6Bvn3bNjA1oLz0TpWZ0ikdvTEoD+6QJKS8mLE8zVDxVzpL9zXusPGVtI/o7mZdPrMA1sX/pppn/1cleBHo4UT241FOkGNkdy1xSdSuycuNLV4IcIuIk7lI8JsyUcnrPhtIkqnd5t7aKv6lMyco9y3JTvMsqacDWz3HOgtphyTJL1sVRsXyajTqPXezhgOZ5jDLOUUFV1f+IJzxrV9qtdqfdf6ZquUyzjW9Le/gAcxGZtg==</latexit>Q✓(s,a)<latexit sha1_base64="0s+my2z5SYqdlyqi30+cKXAoZJs=">AAAC0XicjVHLSsNAFD2Nr1pfVZdugkWoICURQZdFNy5btA9oa5mk0zaYJmEyEUoRxK0/4FZ/SvwD/QvvjCmoRXRCkjPn3nNm7r1O5HuxtKzXjDE3v7C4lF3OrayurW/kN7fqcZgIl9fc0A9F02Ex972A16Qnfd6MBGcjx+cN5/pMxRs3XMReGFzKccQ7IzYIvL7nMknUVbXblkMuWTE+MNl+N1+wSpZe5iywU1BAuiph/gVt9BDCRYIROAJIwj4YYnpasGEhIq6DCXGCkKfjHLfIkTahLE4ZjNhr+g5o10rZgPbKM9Zql07x6RWkNLFHmpDyBGF1mqnjiXZW7G/eE+2p7jamv5N6jYiVGBL7l26a+V+dqkWijxNdg0c1RZpR1bmpS6K7om5ufqlKkkNEnMI9igvCrlZO+2xqTaxrV71lOv6mMxWr9m6am+Bd3ZIGbP8c5yyoH5Zsq2RXjwrl03TUWexgF0Wa5zHKOEcFNfIWeMQTno0LY2zcGfefqUYm1Wzj2zIePgA4VJQa</latexit><latexit sha1_base64="0s+my2z5SYqdlyqi30+cKXAoZJs=">AAAC0XicjVHLSsNAFD2Nr1pfVZdugkWoICURQZdFNy5btA9oa5mk0zaYJmEyEUoRxK0/4FZ/SvwD/QvvjCmoRXRCkjPn3nNm7r1O5HuxtKzXjDE3v7C4lF3OrayurW/kN7fqcZgIl9fc0A9F02Ex972A16Qnfd6MBGcjx+cN5/pMxRs3XMReGFzKccQ7IzYIvL7nMknUVbXblkMuWTE+MNl+N1+wSpZe5iywU1BAuiph/gVt9BDCRYIROAJIwj4YYnpasGEhIq6DCXGCkKfjHLfIkTahLE4ZjNhr+g5o10rZgPbKM9Zql07x6RWkNLFHmpDyBGF1mqnjiXZW7G/eE+2p7jamv5N6jYiVGBL7l26a+V+dqkWijxNdg0c1RZpR1bmpS6K7om5ufqlKkkNEnMI9igvCrlZO+2xqTaxrV71lOv6mMxWr9m6am+Bd3ZIGbP8c5yyoH5Zsq2RXjwrl03TUWexgF0Wa5zHKOEcFNfIWeMQTno0LY2zcGfefqUYm1Wzj2zIePgA4VJQa</latexit><latexit sha1_base64="0s+my2z5SYqdlyqi30+cKXAoZJs=">AAAC0XicjVHLSsNAFD2Nr1pfVZdugkWoICURQZdFNy5btA9oa5mk0zaYJmEyEUoRxK0/4FZ/SvwD/QvvjCmoRXRCkjPn3nNm7r1O5HuxtKzXjDE3v7C4lF3OrayurW/kN7fqcZgIl9fc0A9F02Ex972A16Qnfd6MBGcjx+cN5/pMxRs3XMReGFzKccQ7IzYIvL7nMknUVbXblkMuWTE+MNl+N1+wSpZe5iywU1BAuiph/gVt9BDCRYIROAJIwj4YYnpasGEhIq6DCXGCkKfjHLfIkTahLE4ZjNhr+g5o10rZgPbKM9Zql07x6RWkNLFHmpDyBGF1mqnjiXZW7G/eE+2p7jamv5N6jYiVGBL7l26a+V+dqkWijxNdg0c1RZpR1bmpS6K7om5ufqlKkkNEnMI9igvCrlZO+2xqTaxrV71lOv6mMxWr9m6am+Bd3ZIGbP8c5yyoH5Zsq2RXjwrl03TUWexgF0Wa5zHKOEcFNfIWeMQTno0LY2zcGfefqUYm1Wzj2zIePgA4VJQa</latexit><latexit sha1_base64="0s+my2z5SYqdlyqi30+cKXAoZJs=">AAAC0XicjVHLSsNAFD2Nr1pfVZdugkWoICURQZdFNy5btA9oa5mk0zaYJmEyEUoRxK0/4FZ/SvwD/QvvjCmoRXRCkjPn3nNm7r1O5HuxtKzXjDE3v7C4lF3OrayurW/kN7fqcZgIl9fc0A9F02Ex972A16Qnfd6MBGcjx+cN5/pMxRs3XMReGFzKccQ7IzYIvL7nMknUVbXblkMuWTE+MNl+N1+wSpZe5iywU1BAuiph/gVt9BDCRYIROAJIwj4YYnpasGEhIq6DCXGCkKfjHLfIkTahLE4ZjNhr+g5o10rZgPbKM9Zql07x6RWkNLFHmpDyBGF1mqnjiXZW7G/eE+2p7jamv5N6jYiVGBL7l26a+V+dqkWijxNdg0c1RZpR1bmpS6K7om5ufqlKkkNEnMI9igvCrlZO+2xqTaxrV71lOv6mMxWr9m6am+Bd3ZIGbP8c5yyoH5Zsq2RXjwrl03TUWexgF0Wa5zHKOEcFNfIWeMQTno0LY2zcGfefqUYm1Wzj2zIePgA4VJQa</latexit>s<latexit sha1_base64="O7LJiefM8UmmHRFZga9M5h6iSBo=">AAACxHicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFkUxGUL9gG1SDKd1qFpEmYmQin6A27128Q/0L/wzpiCWkQnJDlz7j1n5t4bppFQ2vNeC87C4tLySnG1tLa+sblV3t5pqSSTjDdZEiWyEwaKRyLmTS10xDup5ME4jHg7HJ2bePuOSyWS+EpPUt4bB8NYDAQLNFENdVOueFXPLnce+DmoIF/1pPyCa/SRgCHDGBwxNOEIARQ9XfjwkBLXw5Q4SUjYOMc9SqTNKItTRkDsiL5D2nVzNqa98VRWzeiUiF5JShcHpEkoTxI2p7k2nllnw/7mPbWe5m4T+oe515hYjVti/9LNMv+rM7VoDHBqaxBUU2oZUx3LXTLbFXNz90tVmhxS4gzuU1wSZlY567NrNcrWbnob2PibzTSs2bM8N8O7uSUN2P85znnQOqr6XtVvHFdqZ/moi9jDPg5pnieo4RJ1NK33I57w7Fw4kaOc7DPVKeSaXXxbzsMHXzyPeA==</latexit><latexit sha1_base64="O7LJiefM8UmmHRFZga9M5h6iSBo=">AAACxHicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFkUxGUL9gG1SDKd1qFpEmYmQin6A27128Q/0L/wzpiCWkQnJDlz7j1n5t4bppFQ2vNeC87C4tLySnG1tLa+sblV3t5pqSSTjDdZEiWyEwaKRyLmTS10xDup5ME4jHg7HJ2bePuOSyWS+EpPUt4bB8NYDAQLNFENdVOueFXPLnce+DmoIF/1pPyCa/SRgCHDGBwxNOEIARQ9XfjwkBLXw5Q4SUjYOMc9SqTNKItTRkDsiL5D2nVzNqa98VRWzeiUiF5JShcHpEkoTxI2p7k2nllnw/7mPbWe5m4T+oe515hYjVti/9LNMv+rM7VoDHBqaxBUU2oZUx3LXTLbFXNz90tVmhxS4gzuU1wSZlY567NrNcrWbnob2PibzTSs2bM8N8O7uSUN2P85znnQOqr6XtVvHFdqZ/moi9jDPg5pnieo4RJ1NK33I57w7Fw4kaOc7DPVKeSaXXxbzsMHXzyPeA==</latexit><latexit sha1_base64="O7LJiefM8UmmHRFZga9M5h6iSBo=">AAACxHicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFkUxGUL9gG1SDKd1qFpEmYmQin6A27128Q/0L/wzpiCWkQnJDlz7j1n5t4bppFQ2vNeC87C4tLySnG1tLa+sblV3t5pqSSTjDdZEiWyEwaKRyLmTS10xDup5ME4jHg7HJ2bePuOSyWS+EpPUt4bB8NYDAQLNFENdVOueFXPLnce+DmoIF/1pPyCa/SRgCHDGBwxNOEIARQ9XfjwkBLXw5Q4SUjYOMc9SqTNKItTRkDsiL5D2nVzNqa98VRWzeiUiF5JShcHpEkoTxI2p7k2nllnw/7mPbWe5m4T+oe515hYjVti/9LNMv+rM7VoDHBqaxBUU2oZUx3LXTLbFXNz90tVmhxS4gzuU1wSZlY567NrNcrWbnob2PibzTSs2bM8N8O7uSUN2P85znnQOqr6XtVvHFdqZ/moi9jDPg5pnieo4RJ1NK33I57w7Fw4kaOc7DPVKeSaXXxbzsMHXzyPeA==</latexit><latexit sha1_base64="O7LJiefM8UmmHRFZga9M5h6iSBo=">AAACxHicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFkUxGUL9gG1SDKd1qFpEmYmQin6A27128Q/0L/wzpiCWkQnJDlz7j1n5t4bppFQ2vNeC87C4tLySnG1tLa+sblV3t5pqSSTjDdZEiWyEwaKRyLmTS10xDup5ME4jHg7HJ2bePuOSyWS+EpPUt4bB8NYDAQLNFENdVOueFXPLnce+DmoIF/1pPyCa/SRgCHDGBwxNOEIARQ9XfjwkBLXw5Q4SUjYOMc9SqTNKItTRkDsiL5D2nVzNqa98VRWzeiUiF5JShcHpEkoTxI2p7k2nllnw/7mPbWe5m4T+oe515hYjVti/9LNMv+rM7VoDHBqaxBUU2oZUx3LXTLbFXNz90tVmhxS4gzuU1wSZlY567NrNcrWbnob2PibzTSs2bM8N8O7uSUN2P85znnQOqr6XtVvHFdqZ/moi9jDPg5pnieo4RJ1NK33I57w7Fw4kaOc7DPVKeSaXXxbzsMHXzyPeA==</latexit>z<latexit sha1_base64="KYpOYd4bXWCbtkGB2ntH6y/Os8E=">AAACxHicjVHLSsNAFD2Nr/quunQTLIKrkoigy6IgLluwD6hFkum0Dk2TMDMRatEfcKvfJv6B/oV3xhTUIjohyZlz7zkz994wjYTSnvdacObmFxaXissrq2vrG5ulre2mSjLJeIMlUSLbYaB4JGLe0EJHvJ1KHozCiLfC4ZmJt265VCKJL/U45d1RMIhFX7BAE1W/uy6VvYpnlzsL/ByUka9aUnrBFXpIwJBhBI4YmnCEAIqeDnx4SInrYkKcJCRsnOMeK6TNKItTRkDskL4D2nVyNqa98VRWzeiUiF5JShf7pEkoTxI2p7k2nllnw/7mPbGe5m5j+oe514hYjRti/9JNM/+rM7Vo9HFiaxBUU2oZUx3LXTLbFXNz90tVmhxS4gzuUVwSZlY57bNrNcrWbnob2PibzTSs2bM8N8O7uSUN2P85zlnQPKz4XsWvH5Wrp/moi9jFHg5onseo4gI1NKz3I57w7Jw7kaOc7DPVKeSaHXxbzsMHb9yPfw==</latexit><latexit sha1_base64="KYpOYd4bXWCbtkGB2ntH6y/Os8E=">AAACxHicjVHLSsNAFD2Nr/quunQTLIKrkoigy6IgLluwD6hFkum0Dk2TMDMRatEfcKvfJv6B/oV3xhTUIjohyZlz7zkz994wjYTSnvdacObmFxaXissrq2vrG5ulre2mSjLJeIMlUSLbYaB4JGLe0EJHvJ1KHozCiLfC4ZmJt265VCKJL/U45d1RMIhFX7BAE1W/uy6VvYpnlzsL/ByUka9aUnrBFXpIwJBhBI4YmnCEAIqeDnx4SInrYkKcJCRsnOMeK6TNKItTRkDskL4D2nVyNqa98VRWzeiUiF5JShf7pEkoTxI2p7k2nllnw/7mPbGe5m5j+oe514hYjRti/9JNM/+rM7Vo9HFiaxBUU2oZUx3LXTLbFXNz90tVmhxS4gzuUVwSZlY57bNrNcrWbnob2PibzTSs2bM8N8O7uSUN2P85zlnQPKz4XsWvH5Wrp/moi9jFHg5onseo4gI1NKz3I57w7Jw7kaOc7DPVKeSaHXxbzsMHb9yPfw==</latexit><latexit sha1_base64="KYpOYd4bXWCbtkGB2ntH6y/Os8E=">AAACxHicjVHLSsNAFD2Nr/quunQTLIKrkoigy6IgLluwD6hFkum0Dk2TMDMRatEfcKvfJv6B/oV3xhTUIjohyZlz7zkz994wjYTSnvdacObmFxaXissrq2vrG5ulre2mSjLJeIMlUSLbYaB4JGLe0EJHvJ1KHozCiLfC4ZmJt265VCKJL/U45d1RMIhFX7BAE1W/uy6VvYpnlzsL/ByUka9aUnrBFXpIwJBhBI4YmnCEAIqeDnx4SInrYkKcJCRsnOMeK6TNKItTRkDskL4D2nVyNqa98VRWzeiUiF5JShf7pEkoTxI2p7k2nllnw/7mPbGe5m5j+oe514hYjRti/9JNM/+rM7Vo9HFiaxBUU2oZUx3LXTLbFXNz90tVmhxS4gzuUVwSZlY57bNrNcrWbnob2PibzTSs2bM8N8O7uSUN2P85zlnQPKz4XsWvH5Wrp/moi9jFHg5onseo4gI1NKz3I57w7Jw7kaOc7DPVKeSaHXxbzsMHb9yPfw==</latexit><latexit sha1_base64="KYpOYd4bXWCbtkGB2ntH6y/Os8E=">AAACxHicjVHLSsNAFD2Nr/quunQTLIKrkoigy6IgLluwD6hFkum0Dk2TMDMRatEfcKvfJv6B/oV3xhTUIjohyZlz7zkz994wjYTSnvdacObmFxaXissrq2vrG5ulre2mSjLJeIMlUSLbYaB4JGLe0EJHvJ1KHozCiLfC4ZmJt265VCKJL/U45d1RMIhFX7BAE1W/uy6VvYpnlzsL/ByUka9aUnrBFXpIwJBhBI4YmnCEAIqeDnx4SInrYkKcJCRsnOMeK6TNKItTRkDskL4D2nVyNqa98VRWzeiUiF5JShf7pEkoTxI2p7k2nllnw/7mPbGe5m5j+oe514hYjRti/9JNM/+rM7Vo9HFiaxBUU2oZUx3LXTLbFXNz90tVmhxS4gzuUVwSZlY57bNrNcrWbnob2PibzTSs2bM8N8O7uSUN2P85zlnQPKz4XsWvH5Wrp/moi9jFHg5onseo4gI1NKz3I57w7Jw7kaOc7DPVKeSaHXxbzsMHb9yPfw==</latexit>HypermodelHidden  Layersf⌫<latexit sha1_base64="ucwl0ZANTGUWXzWAT9fLB1w8Xyo=">AAACynicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFl048JFBfuAtpQkndahaRImE6GE7vwBt/ph4h/oX3hnnIJaRCckOXPuOXfm3usnIU+l47wWrKXlldW14nppY3Nre6e8u9dM40wErBHEYSzavpeykEesIbkMWTsRzJv4IWv540sVb90zkfI4upXThPUm3ijiQx54kqjWsJ93o2zWL1ecqqOXvQhcAyowqx6XX9DFADECZJiAIYIkHMJDSk8HLhwkxPWQEycIcR1nmKFE3oxUjBQesWP6jmjXMWxEe5Uz1e6ATgnpFeS0cUSemHSCsDrN1vFMZ1bsb7lznVPdbUp/3+SaECtxR+xfvrnyvz5Vi8QQ57oGTjUlmlHVBSZLpruibm5/qUpShoQ4hQcUF4QD7Zz32daeVNeueuvp+JtWKlbtA6PN8K5uSQN2f45zETRPqq5TdW9OK7ULM+oiDnCIY5rnGWq4Qh0NXeUjnvBsXVvCmlr5p9QqGM8+vi3r4QNQM5I9</latexit><latexit sha1_base64="ucwl0ZANTGUWXzWAT9fLB1w8Xyo=">AAACynicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFl048JFBfuAtpQkndahaRImE6GE7vwBt/ph4h/oX3hnnIJaRCckOXPuOXfm3usnIU+l47wWrKXlldW14nppY3Nre6e8u9dM40wErBHEYSzavpeykEesIbkMWTsRzJv4IWv540sVb90zkfI4upXThPUm3ijiQx54kqjWsJ93o2zWL1ecqqOXvQhcAyowqx6XX9DFADECZJiAIYIkHMJDSk8HLhwkxPWQEycIcR1nmKFE3oxUjBQesWP6jmjXMWxEe5Uz1e6ATgnpFeS0cUSemHSCsDrN1vFMZ1bsb7lznVPdbUp/3+SaECtxR+xfvrnyvz5Vi8QQ57oGTjUlmlHVBSZLpruibm5/qUpShoQ4hQcUF4QD7Zz32daeVNeueuvp+JtWKlbtA6PN8K5uSQN2f45zETRPqq5TdW9OK7ULM+oiDnCIY5rnGWq4Qh0NXeUjnvBsXVvCmlr5p9QqGM8+vi3r4QNQM5I9</latexit><latexit sha1_base64="ucwl0ZANTGUWXzWAT9fLB1w8Xyo=">AAACynicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFl048JFBfuAtpQkndahaRImE6GE7vwBt/ph4h/oX3hnnIJaRCckOXPuOXfm3usnIU+l47wWrKXlldW14nppY3Nre6e8u9dM40wErBHEYSzavpeykEesIbkMWTsRzJv4IWv540sVb90zkfI4upXThPUm3ijiQx54kqjWsJ93o2zWL1ecqqOXvQhcAyowqx6XX9DFADECZJiAIYIkHMJDSk8HLhwkxPWQEycIcR1nmKFE3oxUjBQesWP6jmjXMWxEe5Uz1e6ATgnpFeS0cUSemHSCsDrN1vFMZ1bsb7lznVPdbUp/3+SaECtxR+xfvrnyvz5Vi8QQ57oGTjUlmlHVBSZLpruibm5/qUpShoQ4hQcUF4QD7Zz32daeVNeueuvp+JtWKlbtA6PN8K5uSQN2f45zETRPqq5TdW9OK7ULM+oiDnCIY5rnGWq4Qh0NXeUjnvBsXVvCmlr5p9QqGM8+vi3r4QNQM5I9</latexit><latexit sha1_base64="ucwl0ZANTGUWXzWAT9fLB1w8Xyo=">AAACynicjVHLSsNAFD2Nr1pfVZdugkVwVRIRdFl048JFBfuAtpQkndahaRImE6GE7vwBt/ph4h/oX3hnnIJaRCckOXPuOXfm3usnIU+l47wWrKXlldW14nppY3Nre6e8u9dM40wErBHEYSzavpeykEesIbkMWTsRzJv4IWv540sVb90zkfI4upXThPUm3ijiQx54kqjWsJ93o2zWL1ecqqOXvQhcAyowqx6XX9DFADECZJiAIYIkHMJDSk8HLhwkxPWQEycIcR1nmKFE3oxUjBQesWP6jmjXMWxEe5Uz1e6ATgnpFeS0cUSemHSCsDrN1vFMZ1bsb7lznVPdbUp/3+SaECtxR+xfvrnyvz5Vi8QQ57oGTjUlmlHVBSZLpruibm5/qUpShoQ4hQcUF4QD7Zz32daeVNeueuvp+JtWKlbtA6PN8K5uSQN2f45zETRPqq5TdW9OK7ULM+oiDnCIY5rnGWq4Qh0NXeUjnvBsXVvCmlr5p9QqGM8+vi3r4QNQM5I9</latexit>randomvector<latexit sha1_base64="FEFfYbLxoKTyCP7yASIRV47YqQc=">AAAC23icjVHLSsNAFD3GV62vqODGTbAIrkoigi6LblxWsFWwRZLpVINJJkwmxVK7cidu/QG3+j/iH+hfeGeMoBbRCUnOnHvPmbn3BmkUZsp1X8as8YnJqenSTHl2bn5h0V5abmYil4w3mIiEPAn8jEdhwhsqVBE/SSX34yDix8Hlvo4f97jMQpEcqX7K27F/noTdkPmKqDN7taX4lVJqIP2kI2Knx5kScnhmV9yqa5YzCrwCVFCsurCf0UIHAgw5YnAkUIQj+MjoOYUHFylxbQyIk4RCE+cYokzanLI4ZfjEXtL3nHanBZvQXntmRs3olIheSUoHG6QRlCcJ69McE8+Ns2Z/8x4YT323Pv2DwismVuGC2L90n5n/1elaFLrYNTWEVFNqGF0dK1xy0xV9c+dLVYocUuI07lBcEmZG+dlnx2gyU7vurW/iryZTs3rPitwcb/qWNGDv5zhHQXOr6rlV73C7UtsrRl3CGtaxSfPcQQ0HqKNB3td4wCOerLZ1Y91adx+p1lihWcG3Zd2/AzwqmT0=</latexit><latexit sha1_base64="FEFfYbLxoKTyCP7yASIRV47YqQc=">AAAC23icjVHLSsNAFD3GV62vqODGTbAIrkoigi6LblxWsFWwRZLpVINJJkwmxVK7cidu/QG3+j/iH+hfeGeMoBbRCUnOnHvPmbn3BmkUZsp1X8as8YnJqenSTHl2bn5h0V5abmYil4w3mIiEPAn8jEdhwhsqVBE/SSX34yDix8Hlvo4f97jMQpEcqX7K27F/noTdkPmKqDN7taX4lVJqIP2kI2Knx5kScnhmV9yqa5YzCrwCVFCsurCf0UIHAgw5YnAkUIQj+MjoOYUHFylxbQyIk4RCE+cYokzanLI4ZfjEXtL3nHanBZvQXntmRs3olIheSUoHG6QRlCcJ69McE8+Ns2Z/8x4YT323Pv2DwismVuGC2L90n5n/1elaFLrYNTWEVFNqGF0dK1xy0xV9c+dLVYocUuI07lBcEmZG+dlnx2gyU7vurW/iryZTs3rPitwcb/qWNGDv5zhHQXOr6rlV73C7UtsrRl3CGtaxSfPcQQ0HqKNB3td4wCOerLZ1Y91adx+p1lihWcG3Zd2/AzwqmT0=</latexit><latexit sha1_base64="FEFfYbLxoKTyCP7yASIRV47YqQc=">AAAC23icjVHLSsNAFD3GV62vqODGTbAIrkoigi6LblxWsFWwRZLpVINJJkwmxVK7cidu/QG3+j/iH+hfeGeMoBbRCUnOnHvPmbn3BmkUZsp1X8as8YnJqenSTHl2bn5h0V5abmYil4w3mIiEPAn8jEdhwhsqVBE/SSX34yDix8Hlvo4f97jMQpEcqX7K27F/noTdkPmKqDN7taX4lVJqIP2kI2Knx5kScnhmV9yqa5YzCrwCVFCsurCf0UIHAgw5YnAkUIQj+MjoOYUHFylxbQyIk4RCE+cYokzanLI4ZfjEXtL3nHanBZvQXntmRs3olIheSUoHG6QRlCcJ69McE8+Ns2Z/8x4YT323Pv2DwismVuGC2L90n5n/1elaFLrYNTWEVFNqGF0dK1xy0xV9c+dLVYocUuI07lBcEmZG+dlnx2gyU7vurW/iryZTs3rPitwcb/qWNGDv5zhHQXOr6rlV73C7UtsrRl3CGtaxSfPcQQ0HqKNB3td4wCOerLZ1Y91adx+p1lihWcG3Zd2/AzwqmT0=</latexit><latexit sha1_base64="FEFfYbLxoKTyCP7yASIRV47YqQc=">AAAC23icjVHLSsNAFD3GV62vqODGTbAIrkoigi6LblxWsFWwRZLpVINJJkwmxVK7cidu/QG3+j/iH+hfeGeMoBbRCUnOnHvPmbn3BmkUZsp1X8as8YnJqenSTHl2bn5h0V5abmYil4w3mIiEPAn8jEdhwhsqVBE/SSX34yDix8Hlvo4f97jMQpEcqX7K27F/noTdkPmKqDN7taX4lVJqIP2kI2Knx5kScnhmV9yqa5YzCrwCVFCsurCf0UIHAgw5YnAkUIQj+MjoOYUHFylxbQyIk4RCE+cYokzanLI4ZfjEXtL3nHanBZvQXntmRs3olIheSUoHG6QRlCcJ69McE8+Ns2Z/8x4YT323Pv2DwismVuGC2L90n5n/1elaFLrYNTWEVFNqGF0dK1xy0xV9c+dLVYocUuI07lBcEmZG+dlnx2gyU7vurW/iryZTs3rPitwcb/qWNGDv5zhHQXOr6rlV73C7UtsrRl3CGtaxSfPcQQ0HqKNB3td4wCOerLZ1Y91adx+p1lihWcG3Zd2/AzwqmT0=</latexit>p(z)<latexit sha1_base64="WPntmJ41jn4URs5Voj+dDzkyZwg=">AAACx3icjVHLSsNAFD2Nr1pfVZdugkWom5KIoMuiG91VsA+oRZJ02gbTTJhMirW48Afc6p+Jf6B/4Z1xCmoRnZDkzLn3nJl7r59EYSod5zVnzc0vLC7llwsrq2vrG8XNrUbKMxGwesAjLlq+l7IojFldhjJirUQwb+hHrOnfnKp4c8REGvL4Uo4T1hl6/TjshYEnFZWU7/aviyWn4uhlzwLXgBLMqvHiC67QBUeADEMwxJCEI3hI6WnDhYOEuA4mxAlCoY4z3KNA2oyyGGV4xN7Qt0+7tmFj2ivPVKsDOiWiV5DSxh5pOOUJwuo0W8cz7azY37wn2lPdbUx/33gNiZUYEPuXbpr5X52qRaKHY11DSDUlmlHVBcYl011RN7e/VCXJISFO4S7FBeFAK6d9trUm1bWr3no6/qYzFav2gcnN8K5uSQN2f45zFjQOKq5TcS8OS9UTM+o8drCLMs3zCFWcoYY6eQ/wiCc8W+cWt0bW7WeqlTOabXxb1sMHrbqQXg==</latexit><latexit sha1_base64="WPntmJ41jn4URs5Voj+dDzkyZwg=">AAACx3icjVHLSsNAFD2Nr1pfVZdugkWom5KIoMuiG91VsA+oRZJ02gbTTJhMirW48Afc6p+Jf6B/4Z1xCmoRnZDkzLn3nJl7r59EYSod5zVnzc0vLC7llwsrq2vrG8XNrUbKMxGwesAjLlq+l7IojFldhjJirUQwb+hHrOnfnKp4c8REGvL4Uo4T1hl6/TjshYEnFZWU7/aviyWn4uhlzwLXgBLMqvHiC67QBUeADEMwxJCEI3hI6WnDhYOEuA4mxAlCoY4z3KNA2oyyGGV4xN7Qt0+7tmFj2ivPVKsDOiWiV5DSxh5pOOUJwuo0W8cz7azY37wn2lPdbUx/33gNiZUYEPuXbpr5X52qRaKHY11DSDUlmlHVBcYl011RN7e/VCXJISFO4S7FBeFAK6d9trUm1bWr3no6/qYzFav2gcnN8K5uSQN2f45zFjQOKq5TcS8OS9UTM+o8drCLMs3zCFWcoYY6eQ/wiCc8W+cWt0bW7WeqlTOabXxb1sMHrbqQXg==</latexit><latexit sha1_base64="WPntmJ41jn4URs5Voj+dDzkyZwg=">AAACx3icjVHLSsNAFD2Nr1pfVZdugkWom5KIoMuiG91VsA+oRZJ02gbTTJhMirW48Afc6p+Jf6B/4Z1xCmoRnZDkzLn3nJl7r59EYSod5zVnzc0vLC7llwsrq2vrG8XNrUbKMxGwesAjLlq+l7IojFldhjJirUQwb+hHrOnfnKp4c8REGvL4Uo4T1hl6/TjshYEnFZWU7/aviyWn4uhlzwLXgBLMqvHiC67QBUeADEMwxJCEI3hI6WnDhYOEuA4mxAlCoY4z3KNA2oyyGGV4xN7Qt0+7tmFj2ivPVKsDOiWiV5DSxh5pOOUJwuo0W8cz7azY37wn2lPdbUx/33gNiZUYEPuXbpr5X52qRaKHY11DSDUlmlHVBcYl011RN7e/VCXJISFO4S7FBeFAK6d9trUm1bWr3no6/qYzFav2gcnN8K5uSQN2f45zFjQOKq5TcS8OS9UTM+o8drCLMs3zCFWcoYY6eQ/wiCc8W+cWt0bW7WeqlTOabXxb1sMHrbqQXg==</latexit><latexit sha1_base64="WPntmJ41jn4URs5Voj+dDzkyZwg=">AAACx3icjVHLSsNAFD2Nr1pfVZdugkWom5KIoMuiG91VsA+oRZJ02gbTTJhMirW48Afc6p+Jf6B/4Z1xCmoRnZDkzLn3nJl7r59EYSod5zVnzc0vLC7llwsrq2vrG8XNrUbKMxGwesAjLlq+l7IojFldhjJirUQwb+hHrOnfnKp4c8REGvL4Uo4T1hl6/TjshYEnFZWU7/aviyWn4uhlzwLXgBLMqvHiC67QBUeADEMwxJCEI3hI6WnDhYOEuA4mxAlCoY4z3KNA2oyyGGV4xN7Qt0+7tmFj2ivPVKsDOiWiV5DSxh5pOOUJwuo0W8cz7azY37wn2lPdbUx/33gNiZUYEPuXbpr5X52qRaKHY11DSDUlmlHVBcYl011RN7e/VCXJISFO4S7FBeFAK6d9trUm1bWr3no6/qYzFav2gcnN8K5uSQN2f45zFjQOKq5TcS8OS9UTM+o8drCLMs3zCFWcoYY6eQ/wiCc8W+cWt0bW7WeqlTOabXxb1sMHrbqQXg==</latexit>Predict Layer  ✓hidden<latexit sha1_base64="eOy4VjAcjqHV4oY7ntK7seaJ9Fg=">AAAC4HicjVHLSsNAFD2Nr1pfVZfdBIvgqiRudFlw47KCfUBTSpJO26F5kUzEUrpw507c+gNudeOviH+gf+GdaQpqEZ2Q5My555yZO+NEHk+EYbzltKXlldW1/HphY3Nre6e4u9dIwjR2Wd0NvTBuOXbCPB6wuuDCY60oZrbveKzpjM5kvXnF4oSHwaUYR6zj24OA97lrC6K6xZIVDXl3YokhE3bXEuxaTIa812PBdNotlo2KoYa+CMwMlKv6y7IOoBYWX2GhhxAuUvhgCCAIe7CR0NOGCQMRcR1MiIsJcVVnmKJA3pRUjBQ2sSP6DmjWztiA5jIzUW6XVvHojcmp45A8IeliwnI1XdVTlSzZ37InKlPubUx/J8vyiRUYEvuXb678r0/2ItDHqeqBU0+RYmR3bpaSqlORO9e/dCUoISJO4h7VY8Kucs7PWVeeRPUuz9ZW9XellKycu5k2xYfcJV2w+fM6F0HjuGIaFfPCLFermI08SjjAEd3nCao4Rw11yr7BI57wrDnarXan3c+kWi7z7OPb0B4+AdDDnLk=</latexit>Base model ✓predict<latexit sha1_base64="o82LYeYO7uabQJENMveYb+0TTbo=">AAAC2nicjVHLSsNAFD2Nr1pfVXHlJrQIrkriRpcFNy4r2Ae0pSTTaTuYJiGZiKV0I27ErT/QrX6Q+Af6F96ZpqAW0RuSnDn3njNz57qhJ2JpWW8ZY2l5ZXUtu57b2Nza3snv7tXiIIkYr7LAC6KG68TcEz6vSiE93ggj7gxdj9fd63OVr9/wKBaBfyVHIW8Pnb4veoI5kqhO/qAlB1w6nZbkt3JM0q5gctLJF62SpcNcBHYKiuXC9B4UlSD/iha6CMCQYAgOH5KwBwcxPU3YsBAS18aYuIiQ0HmOCXKkTaiKU4VD7DV9+7RqpqxPa+UZazWjXTx6I1KaOCJNQHURYbWbqfOJdlbsb95j7anONqK/m3oNiZUYEPuXbl75X53qRaKHM92DoJ5CzajuWOqS6FtRJze/dCXJISRO4S7lI8JMK+f3bGpNrHtXd+vo/LuuVKxas7Q2wYc6JQ3Y/jnORVA7KdlWyb60i+UyZpHFIQo4pnmeoowLVFAl7zGmeMaL0TLujAfjcVZqZFLNPr6F8fQJhumaVQ==</latexit> ✓predict:=f⌫(z)<latexit sha1_base64="Cf7ORX84wKoQ3f+KBMGFLVYBdgs=">AAAC6nicjVHLTttAFD1xXyG0JS1LNhYRUrqJ7G5AlZCisukySIRESlBkTyYwwrEte1xBo3wBO3aoW36ALXxI1T+ALV/AmcGR2qKqvZbtM+fec2bu3DCNVK4972fFefb8xctX1aXa8us3b1fq797v50mRCdkVSZRk/TDIZaRi2dVKR7KfZjKYhpHshcc7Jt/7KrNcJfGePk3lwTQ4jNVEiUCTGtU3hvpI6mA0G2p5omfUjpXQ87n7adudkI2LefPbh1G94bU8G+5T4Jeg0V6+/wxGJ6n/wBBjJBAoMIVEDE0cIUDOZwAfHlJyB5iRy4iUzUvMUaO2YJVkRUD2mN9DrgYlG3NtPHOrFtwl4ptR6WKDmoR1GbHZzbX5wjob9m/eM+tpznbKf1h6TclqHJH9l25R+b8604vGBFu2B8WeUsuY7kTpUthbMSd3f+lK0yElZ/CY+YxYWOXinl2ryW3v5m4Dm7+1lYY1a1HWFrgzp+SA/T/H+RTsf2z5Xsvf9RvtJh6jijWso8l5bqKNL+igS+8zXOEaN07knDsXzvfHUqdSalbxWziXD39SoEs=</latexit>p(✓predict)<latexit sha1_base64="TcrJd+3RQDzdYNzboKsEf0Xgwxc=">AAAC33icjVHLSsNAFD2Nr1pfVXe6CRahbkoigi5FNy4r2Ae0pUymow6mSUgmopSCO3fi1h9wq38j/oH+hXfGFNQiOiHJmXPvOTP3Xi/yZaIc5zVnTUxOTc/kZwtz8wuLS8XllXoSpjEXNR76Ydz0WCJ8GYiaksoXzSgWrO/5ouFdHOp441LEiQyDE3UdiU6fnQXyVHKmiOoW16JyW50LxbqDthJXakDqnuRqONzqFktOxTHLHgduBkrIVjUsvqCNHkJwpOhDIIAi7IMhoacFFw4i4joYEBcTkiYuMESBtCllCcpgxF7Q94x2rYwNaK89E6PmdIpPb0xKG5ukCSkvJqxPs008Nc6a/c17YDz13a7p72VefWIVzon9SzfK/K9O16Jwij1Tg6SaIsPo6njmkpqu6JvbX6pS5BARp3GP4jFhbpSjPttGk5jadW+Zib+ZTM3qPc9yU7zrW9KA3Z/jHAf17YrrVNzjndL+QTbqPNaxgTLNcxf7OEIVNfK+wSOe8Gwx69a6s+4/U61cplnFt2U9fAAo6Jq+</latexit><latexit sha1_base64="TcrJd+3RQDzdYNzboKsEf0Xgwxc=">AAAC33icjVHLSsNAFD2Nr1pfVXe6CRahbkoigi5FNy4r2Ae0pUymow6mSUgmopSCO3fi1h9wq38j/oH+hXfGFNQiOiHJmXPvOTP3Xi/yZaIc5zVnTUxOTc/kZwtz8wuLS8XllXoSpjEXNR76Ydz0WCJ8GYiaksoXzSgWrO/5ouFdHOp441LEiQyDE3UdiU6fnQXyVHKmiOoW16JyW50LxbqDthJXakDqnuRqONzqFktOxTHLHgduBkrIVjUsvqCNHkJwpOhDIIAi7IMhoacFFw4i4joYEBcTkiYuMESBtCllCcpgxF7Q94x2rYwNaK89E6PmdIpPb0xKG5ukCSkvJqxPs008Nc6a/c17YDz13a7p72VefWIVzon9SzfK/K9O16Jwij1Tg6SaIsPo6njmkpqu6JvbX6pS5BARp3GP4jFhbpSjPttGk5jadW+Zib+ZTM3qPc9yU7zrW9KA3Z/jHAf17YrrVNzjndL+QTbqPNaxgTLNcxf7OEIVNfK+wSOe8Gwx69a6s+4/U61cplnFt2U9fAAo6Jq+</latexit><latexit sha1_base64="TcrJd+3RQDzdYNzboKsEf0Xgwxc=">AAAC33icjVHLSsNAFD2Nr1pfVXe6CRahbkoigi5FNy4r2Ae0pUymow6mSUgmopSCO3fi1h9wq38j/oH+hXfGFNQiOiHJmXPvOTP3Xi/yZaIc5zVnTUxOTc/kZwtz8wuLS8XllXoSpjEXNR76Ydz0WCJ8GYiaksoXzSgWrO/5ouFdHOp441LEiQyDE3UdiU6fnQXyVHKmiOoW16JyW50LxbqDthJXakDqnuRqONzqFktOxTHLHgduBkrIVjUsvqCNHkJwpOhDIIAi7IMhoacFFw4i4joYEBcTkiYuMESBtCllCcpgxF7Q94x2rYwNaK89E6PmdIpPb0xKG5ukCSkvJqxPs008Nc6a/c17YDz13a7p72VefWIVzon9SzfK/K9O16Jwij1Tg6SaIsPo6njmkpqu6JvbX6pS5BARp3GP4jFhbpSjPttGk5jadW+Zib+ZTM3qPc9yU7zrW9KA3Z/jHAf17YrrVNzjndL+QTbqPNaxgTLNcxf7OEIVNfK+wSOe8Gwx69a6s+4/U61cplnFt2U9fAAo6Jq+</latexit><latexit sha1_base64="TcrJd+3RQDzdYNzboKsEf0Xgwxc=">AAAC33icjVHLSsNAFD2Nr1pfVXe6CRahbkoigi5FNy4r2Ae0pUymow6mSUgmopSCO3fi1h9wq38j/oH+hXfGFNQiOiHJmXPvOTP3Xi/yZaIc5zVnTUxOTc/kZwtz8wuLS8XllXoSpjEXNR76Ydz0WCJ8GYiaksoXzSgWrO/5ouFdHOp441LEiQyDE3UdiU6fnQXyVHKmiOoW16JyW50LxbqDthJXakDqnuRqONzqFktOxTHLHgduBkrIVjUsvqCNHkJwpOhDIIAi7IMhoacFFw4i4joYEBcTkiYuMESBtCllCcpgxF7Q94x2rYwNaK89E6PmdIpPb0xKG5ukCSkvJqxPs008Nc6a/c17YDz13a7p72VefWIVzon9SzfK/K9O16Jwij1Tg6SaIsPo6njmkpqu6JvbX6pS5BARp3GP4jFhbpSjPttGk5jadW+Zib+ZTM3qPc9yU7zrW9KA3Z/jHAf17YrrVNzjndL+QTbqPNaxgTLNcxf7OEIVNfK+wSOe8Gwx69a6s+4/U61cplnFt2U9fAAo6Jq+</latexit>Published as a conference paper at ICLR 2022

• Second, the feature mapping ϕθhidden(·) is updated in an end-to-end way during the training. This

deals with the feature learning problem of RLSVI (i.e., issue 1).

• Third, the posterior samples of θpredict are obtained without involving (1.2). Specifically, the
hypermodel could directly output an approximate posterior sample by taking z as its input. This
addresses the computational problem of RLSVI when the feature is changing (i.e., issue 2).

• Finally, compared with the finite ensembles in BootDQN, our approach learns the posterior
distribution in a meta way and the hypermodel has certain generalization ability. Thus, our
approach could have a better posterior approximation with the same computation resources. As
such, our method is more computationally efficient to achieve a near-optimal performance.

To evaluate the efficiency of algorithms, we first consider the Atari suite (Bellemare et al., 2013) and
assess the efficiency in terms of the human-normalized score over 49 tasks. The empirical result
suggests HyperDQN with 20M frames outperforms DQN (Mnih et al., 2015) with 200M frames in
terms of the maximum human-normalized score. With the same training frames, HyperDQN also
outperforms the exploration bonus methods OPIQ (Rashid et al., 2020) and OB2I (Bai et al., 2021),
and randomized exploration methods BootDQN (Osband et al., 2018) and NoisyNet (Fortunato et al.,
2018). For another challenging benchmark SuperMarioBros (Kauten, 2018), HyperDQN beats these
baselines on 5 out of 9 games in terms of the raw scores.

2 BACKGROUND

Markov Decision Process. In the standard reinforcement learning (RL) framework (Sutton &
Barto, 2018), a learning agent interacts with an Markov Decision Process (MDP) to improve its
performance via maximizing cumulative reward. The sequential decision process is characterized
as follows: at each timestep t, the agent receives a state st from the environment and selects an
action at from its policy π(a|s) = Pr{a = at|s = st}; this decision is sent back to the environment,
and the environment gives a reward signal r(st, at) and transits to the next state st+1 based on the
s,s′ = Pr{s′ = st+1|s = st, a = at}. The main target of RL is to
state transition probability pa
(cid:12)
maximize the (expected) discounted return E(cid:2) (cid:80)∞
t=0 γtr(st, at)
(cid:12)s0 ∼ ρ(·)], where ρ(·) is initial state
distribution and γ ∈ (0, 1) is a discount factor.

Deep Q-Networks (DQN). In Deep Q-Networks (DQN) (Mnih et al., 2015), it employs a neural net-
work to approximate the Q-value function, which is defined as Qπ(s, a) = E[(cid:80)∞
t=0 γtr(st, at)|s0 =
s, a0 = a]. In particular, a temporal difference (TD) based objective is applied:
Qθ(s′, a′) − Qθ(s, a)(cid:1)2

(2.1)

(cid:88)

(cid:0)r(s, a) + γ max
a′

min
θ

,

(s,a,r,s′)∼D

where D is the experience replay buffer, Qθ is a prediction network parameterized by θ, and Qθ is
the so-called target network, which is a delayed copy of Qθ for stable training.

3 RELATED WORK

Epistemic uncertainty qualification. It is justified that dithering strategies like ϵ-greedy (Mnih
et al., 2015) are inefficient (Osband et al., 2019). This is intuitive since they do not have any
epistemic uncertainty measure and hence cannot “write-off” sub-optimal actions after experimentation.
Importantly, epistemic uncertainty reflects the confidence about the unknown environment in online
decision-making (Osband, 2016a). For tabular MDPs with the one-hot feature, “count” provides an
epistemic uncertainty measure (Brafman & Tennenholtz, 2002; Jaksch et al., 2010; Azar et al., 2017;
Jin et al., 2018) and “covariance” serves as the counterpart for linear MDPs (Jin et al., 2020; Cai
et al., 2020). Currently, we do not have a perfect epistemic uncertainty qualification tool for deep RL,
where a good feature is unknown in advance. As argued in (Osband et al., 2018), approaches like
dropout (Srivastava et al., 2014) and distributional operator (Bellemare et al., 2017) are not suitable
since they are designed for risk estimation. In this paper, we focus on the extension of hypermodel
(Dwaracherla et al., 2020), which could capture the epistemic uncertainty for bandit tasks.

OFU-based exploration. Based on the mentioned uncertainty qualification, optimism in the face of
uncertainty (OFU) based methods (Jaksch et al., 2010; Azar et al., 2017; Jin et al., 2018) construct
upper confidence bound (UCB) to direct exploration. These theoretical works have inspired many
empirical studies that encourage exploration by adding “reward/exploration bonus” to mimic UCB
(Stadie et al., 2015; Pathak et al., 2017; Tang et al., 2017; Burda et al., 2019a;b). The challenges in this

3

Published as a conference paper at ICLR 2022

direction include: how to obtain task-relevant reward bonus (O’Donoghue et al., 2018)? how to get
an optimistic initialization with neural network (Rashid et al., 2020)? how to properly back-propagate
the uncertainty over periods to induce temporally extended behaviors (Bai et al., 2021)? Though some
of SOTA exploration bonus methods have achieved superior performance on some hard exploration
tasks, (Ta¨ıga et al., 2020) report that these algorithms do not provide meaningful gains over the
ϵ-greedy scheme on the whole Atari suite.

TS-based exploration. On the other hand, Thompson sampling (TS) based methods design the
exploration strategy in a Bayesian way (Osband et al., 2013; Osband & Van Roy, 2017). Randomized
least-square value iteration (RLSVI) provides a promising direction (Osband et al., 2016b; 2019). As
we have mentioned in Section 1, this method, however, can only be applied when a good feature is
known and fixed during the training. Following RLSVI, Osband et al. (2016a) develop a practical
algorithm called Bootstrapped DQN, which uses finite ensembles to generate the randomized value
functions. As discussed earlier, the main bottleneck of Bootstrapped DQN is its computation
complexity: it requires lots of ensembles to obtain an accurate approximation of the posterior samples
(Lu & Van Roy, 2017). Our method is closely related to NoisyNet (Fortunato et al., 2018) as both
methods inject noise to the parameter. However, we cannot view NoisyNet as an extension of RLSVI.
The main reason is that NoisyNet is not ensured to approximate the posterior distribution as stated
in (Fortunato et al., 2018). Besides, Ishfaq et al. (2021) attempt to extend the idea of RLSVI to the
general case. In particular, their algorithm solves M randomized least-square problems and chooses
the most optimistic value function in each iteration. Since M is selected to be proportional to the
inherent dimension in theory, their algorithm is also not computationally efficient.

4 METHODOLOGY

4.1 ARCHITECTURE DESIGN

In this part, we explain the architecture design of HyperDQN shown in Figure 1. First, the hypermodel
fν(·) maps a Nz-dimension random vector z ∼ p(z) (e.g., a Gaussian distribution) to a specific
d-dimension parameter θ := fν(z). For example, if z follows an isotropic Gaussian distribution
N (0, I) and the hypermodel fν(z) = ν⊤
w νw),
where νw ∈ RNz×d and νb ∈ Rd are the weight and bias of the linear hypermodel with ν = (νw, νb).
In general, the hypermodel could be a non-linear mapping so that fν(z) could follow an arbitrary
distribution. Unless stated otherwise, we only consider the linear hypermodel rather than non-linear
ones. This is because linear hypermodel has sufficient representation power to approximate the
posterior distribution (like RLSVI does) (Dwaracherla et al. (2020)).

w z + νb is a linear model, then θ := fν(z) ∼ N (νb, ν⊤

Now, we explain the basemodel in Figure 1. The base model (i.e., a deep neural network) operates in
a standard way: it takes a state s as its input and outputs Qθ(s, a) ∈ R for some action a. Concretely,
the base model employs a feature extractor ϕθhidden : S → Rd (i.e., the hidden layers in Figure 1)
to learn a good representation. Then, the base model outputs Q(s, a) by Q(s, ·) = θ⊤
predictϕθhidden (s),
where θpredict ∈ Rd×Na is the parameter of the prediction layer (i.e., the last layer of the base
model in Figure 1). By our formulation, θpredict is also the output of the hypermodel. That is,
θpredict = fν(z) + fνprior (z), where fνprior (z) is a fixed prior hypermodel; see the discussion below.
Numerically, we fix vprior = v0, where v0 is a random initialization. In the sequel, we will introduce
the difference between the hypermodel in Figure 1 and that in Dwaracherla et al. (2020).

4.2 DIFFERENCES WITH DWARACHERLA ET AL. (2020): HOW THE DIRECT EXTENSION FAILS

Readers may notice that our architecture design is different from the original one in (Dwaracherla et al.,
2020). Concretely, we apply the hypermodel for the last layer of the base model, but in (Dwaracherla
et al., 2020) the hypermodel is applied for all layers of the base model. This modification is aimed
to overcome the training difficulty: the training will fail if otherwise (see Appendix E.2 for the
numerical evidence). Why does the training fail under the original design? We find out it is due
to the severe gradient explosion at the initialization, which does not happen if no hypermodel is
attached. For standard deep neural network models, such gradient explosion is mainly avoided by
carefully designed initialization strategies (Sun, 2019). For instance, the famous LeCun initialization
rule (LeCun et al., 1998) suggests that we should initialize the parameter of i-th layer by sampling
from the Gaussian distribution N (0, σ2) with σ = 1/(cid:112)di−1, where di−1 is the width of (i − 1)-th
layer. As a result, the input signal and the output signal have the same order ℓ2-norm after processing.

4

Published as a conference paper at ICLR 2022

However, things have changed if we directly use the architecture in (Dwaracherla et al., 2020) with
this initialization technique: the output of the hypermodel is expected to lie between −1 and 1 (up
to constants); this further implies that the parameter is in [−1, 1] for each layer of the base model,
which disobeys the principle 1/(cid:112)di−1. Consequently, the input signal amplifies over layers and the
gradient explodes. To bypass the training issue, Dwaracherla et al. (2020) use a shadow base model
(e.g., 2 layers with the width of 10) and further implement the blocked hypermodel; see (Dwaracherla
et al., 2020, Section 4) for details. But for deep RL, this training issue is severe and we overcome
this challenge by the proposed architecture shown in Figure 1, in which the hidden layers of the base
model are initialized with common techniques and the last layer could be properly initialized by
normalizing the output of the hypermodel. This deals with the mentioned parameter initialization
issue. Fortunately, this simple architecture still retains the main ingredient of RLSVI (i.e., capturing
the posterior distribution over a linear prediction function).

4.3 TRAINING OBJECTIVE
In this part, we introduce the objective function. For the sake of better presentation, let us first
consider a regression task. Let x ∈ Rd be the feature and y ∈ R be the label, the objective function
developed in (Dwaracherla et al., 2020) for training the hypermodel is

L(ν; D) =

(cid:90)

z

(cid:20) (cid:88)

p(z)

(x,y,ξ)∈D

(cid:0)y + σωz⊤ξ
(cid:124) (cid:123)(cid:122) (cid:125)
(a)

− (gfνprior (z)(x) + gfν (z)(x))
(cid:123)(cid:122)
(cid:125)
(b)

(cid:124)

(cid:1)2

+

σ2
ω
σ2
p
(cid:124)

∥fν(z)∥2

(cid:21)

(dz),

(cid:123)(cid:122)
(c)

(cid:125)

(4.1)
where ξ is a random vector independently sampled from the unit hypersphere, paired with each
(x, y) in the dataset D = {(xi, yi, ξi) : i = 1, . . . , |D|}; σω > 0 is the noise scale and σp > 0 is
the regularization scale. Next, we briefly explain three key parts in (4.1). First, (a) is an artificial
noise term exerted on the label y, which is introduced for pure technical purpose. We defer more
discussion of (a) to Remark 1. Second, (b) contains two base models denoted by g. Concretely,
gfνprior (z)(·) is a prior model parameterized by the output of a fixed hypermodel fνprior. This term
stores the prior information. At the same time, gfν (z)(·) is a differential model linked by the output
of a trainable hypermodel fν. We remark that term (b) is called “additive prior models” in (Osband
et al., 2018; Dwaracherla et al., 2020); see (Dwaracherla et al., 2020, Section 2.5) for a detailed
discussion. Finally, (c) is a regularization term, which is an essential design in Bayesian learning.

To address the limitations of RLSVI for deep RL tasks, we first employ a feature mapping (param-
eterized by θhidden) and approximate the Q-value function by a linear function (parameterized by
θpredict) over the learned feature. Furthermore, the parameter of this linear function is modeled by
the hypermodel (i.e., θpredict = fνprior (z) + fν(z)) as discussed in Section 4.1. To this end, we extend
(4.1) and implement the following optimization problem:

min
ν,θhidden

(cid:90)

z

(cid:20) (cid:88)

p(z)

(s,a,r,ξ,s′)∈D

(cid:0)Qtarget(s′, z) + σωz⊤ξ − Qprediction(s, a, z)(cid:1)2

+

∥fν(z)∥2

(cid:21)

(dz),

σ2
ω
σ2
p

where

Qprediction(s, a, z) = Qθprior,fνprior (z)(s, a) + Qθhidden,fν (z)(s, a),

Qtarget(s′, z) = r + γ max
a′

(cid:104)

(cid:105)
Qθprior,fνprior (z)(s′, a′) + Q¯θhidden,f¯ν (z)(s′, a′)

.

(4.2)

(4.3)

Similar to the formulation in the regression problem, the Q-value function is the sum of the prior
Qθprior,fνprior (z) and the differential Qθhidden,fν (z). Following the target network in DQN, (θhidden, ν) is
the delayed copy of (θhidden, ν). Compared with the TD loss in (2.1), there is an additional noise
term σωz⊤ξ for Bayesian learning and the prediction layer of the Q-value function is modeled as a
probabilistic layer by the hypermodel. In experiments, we use the average of finite samples of z to
approximate (4.2); see the empirical loss function in (A.6) in Appendix. We choose Adam (Kingma
& Ba, 2015) to optimize the empirical loss function.

4.4 THEORETICAL GUARANTEE
To understand our method, we explain why (4.1) is a good objective to generate approximate posterior
samples. We provide the analysis under the linear case.

5

Published as a conference paper at ICLR 2022

Assumption 1. Suppose the data generation follows y = x⊤θ⋆ + ω⋆, ω⋆ ∼ N (0, σ2
ω) and the prior
distribution over θ⋆ is N (θp, σ2
pI). Furthermore, assume the base model is linear, i.e., gfν (z)(x) =
x⊤fν(z) and gfνprior (z)(x) = x⊤fνprior (z). Moreover, assume the hypermodel is also linear, i.e.,
fν(z) = ν⊤

; ν = (νw, νb) and νprior = (νprior

w z + νb and fνprior (z) = νprior⊤

w , νprior

z + νprior
b

).

w

b

Based on Assumption 1, the posterior distribution over θ⋆ is N (E[θ⋆ | D], Cov[θ⋆ | D]) with

(cid:18) 1
σ2
ω
(cid:18) 1
σ2
ω

E[θ⋆ | D] =

X ⊤X +

1
σ2
p

(cid:19)−1 (cid:18) 1
σ2
ω

I

X ⊤Y +

(cid:19)

θp

,

1
σ2
p

(cid:19)−1

I

X ⊤X +

Cov[θ⋆ | D] =

1
σ2
p
where X ∈ R|D|×d and Y ∈ R|D| are concatenation of xi and yi, respectively. Also, we have that
gfνprior (z)(x) + gfν (z)(x) = gfνprior (z)+fν (z)(x) = gθ(x) = x⊤θ, where θ := fνprior (z) + fν(z).
Theorem 1 (Informal). Under Assumption 1, set νprior
b ) be
the optimal solution of (4.1) conditioned on specific realizations of ξ , then θ := fνprior (z) + fν⋆ (z) ∼
N (νprior

b = θp. Let ν⋆ = (ν⋆

w = σpI and νprior

w, ν⋆

,

b , (νprior

b + ν⋆
νprior
b + ν⋆

w + ν⋆
b = E[θ⋆ | D],
Furthermore, the error term satisfies Eξ[err(ξ)] = 0.

w + ν⋆
(νprior

w)) satisfies

w + ν⋆

w)⊤(νprior

w)⊤(νprior

w + ν⋆

w) = Cov[θ⋆ | D] + err(ξ).

Theorem 1 states that the optimized hypermodel can approx-
imate the true posterior distribution. The formal statement
and proof are given in Appendix B.3. We emphasize that the
message in Theorem 1 is not discussed in (Dwaracherla et al.,
2020, Theorem 1). Dwaracherla et al. (2020) prove that a linear
hypermodel has sufficient representation power to approximate
any distribution (over functions) so it is unnecessary to use a
non-linear one. However, this guarantee is unrelated to the ob-
jective (4.1): Dwaracherla et al. (2020) only show there exists
a linear hypermodel can approximate the posterior distribution
but Dwaracherla et al. (2020) do not tell us whether (4.1) can
lead to such a hypermodel. Instead, this question is affirma-
tively answered by Theorem 1. In addition, Theorem 1 also
conveys an important message about the noise used in (4.1), which we explain in Remark 1.

Figure 2: Visualization of true pos-
terior samples and learned posterior
samples.

Remark 1. Readers may ask whether an independent Gaussian noise ω (rather than the z-dependent
noise z⊤ξ) is applicable in (4.1). Intuitively, ω can also form a randomized-least square problem to
optimize the hypermodel. However, by inspecting the proof of Theorem 1, we claim that this scheme
does not work. Technically speaking, ω will introduce exogenous random sources into the objective
function, which jeopardizes the learning goal of mapping an index z to a posterior sample θ; see
Appendix B.4 for a formal argument.

To better understand the fundamental issue involved here, we run simulation on a toy 2-dimensional
Bayesian linear regression problem (Assumption 1) with θ⋆ = (θ⋆
b ); see Appendix C.3 for
experiment details. After optimizing (4.1), we visualize the posterior samples generated by the linear
hypermodel in Figure 2. In particular, hypermodel(z⊤ξ) (in green) and hypermodel(ω) (in
red) correspond to learned samples via solving (4.1) with a z-dependent noise z⊤ξ and an independent
Gaussian noise ω, respectively. The variant posterior sample (in purple) is obtain by the
true posterior distribution N (E[θ⋆ | D], Cov[θ⋆ | D]). We observe that the z-dependent noise is
indispensable for the hypermodel to approximate the posterior distribution.

w, θ⋆

4.5 THE ALGORITHM

We outline the proposed method in Algorithm 2 in Appendix. At the beginning of each episode, a
random vector z is sampled to obtain the Q-value function, which is further used for interactions with
environments. Subsequently, the randomized temporal difference objective in (4.2) is optimized to
train the hypermodel and the hidden layers based on the collected data.

6

 ✓?w<latexit sha1_base64="5dSD+hpxhEQXI1QGjsAKxf0iVoM=">AAAC0XicjVHNSsNAGJzGv1r/ql4EL8EieCqJB+2x4MWjoq2FtpZNum2DaRJ2N0oRQbz6Al71SXwL8Q30Lfx2TUEtohuSzM43M7vfrpeEgVSO85qzpqZnZufy84WFxaXlleLqWl3GqfB5zY/DWDQ8JnkYRLymAhXyRiI4G3ohP/MuDnT97JILGcTRqRolvD1k/SjoBT5TRJ231IAr1rk6b0nFRKdYcsqOGfYkcDNQqm7sqWcAR3HxBS10EcNHiiE4IijCIRgkPU24cJAQ18Y1cYJQYOocNyiQNyUVJwUj9oK+fZo1Mzaiuc6Uxu3TKiG9gpw2tskTk04Q1qvZpp6aZM3+ln1tMvXeRvT3sqwhsQoDYv/yjZX/9eleFHqomB4C6ikxjO7Oz1JScyp65/aXrhQlJMRp3KW6IOwb5/icbeORpnd9tszU34xSs3ruZ9oU73qXdMHuz+ucBPXdsuuU3WO3VK3gc+SxiS3s0H3uo4pDHKFG2QIPeMSTdWKNrFvr7lNq5TLPOr4N6/4D25GW+A==</latexit>✓?b<latexit sha1_base64="8rA1zy3wraoZ54fWw8n59vFsH9U=">AAAC13icjVHLSsNAFD2Nr1pftS7dBIvgqiQi6LLoxmUF+xBbyySdtqFpEiYTsZTiTtz6A271j8Q/0L/wzpiCWkQnJDlz7j1n5t7rRL4XS8t6zRhz8wuLS9nl3Mrq2vpGfrNQi8NEuLzqhn4oGg6Lue8FvCo96fNGJDgbOj6vO4MTFa9fcxF7YXAuRxFvDVkv8LqeyyRR7XyhKftcsvbYmVyNm7FkYtLOF62SpZc5C+wUFJGuSph/QRMdhHCRYAiOAJKwD4aYnkvYsBAR18KYOEHI03GOCXKkTSiLUwYjdkDfHu0uUzagvfKMtdqlU3x6BSlN7JImpDxBWJ1m6niinRX7m/dYe6q7jejvpF5DYiX6xP6lm2b+V6dqkejiSNfgUU2RZlR1buqS6K6om5tfqpLkEBGncIfigrCrldM+m1oT69pVb5mOv+lMxaq9m+YmeFe3pAHbP8c5C2r7Jdsq2WcHxfJxOuostrGDPZrnIco4RQVV8r7BI57wbFwYt8adcf+ZamRSzRa+LePhA6qxl3c=</latexit><latexit sha1_base64="8rA1zy3wraoZ54fWw8n59vFsH9U=">AAAC13icjVHLSsNAFD2Nr1pftS7dBIvgqiQi6LLoxmUF+xBbyySdtqFpEiYTsZTiTtz6A271j8Q/0L/wzpiCWkQnJDlz7j1n5t7rRL4XS8t6zRhz8wuLS9nl3Mrq2vpGfrNQi8NEuLzqhn4oGg6Lue8FvCo96fNGJDgbOj6vO4MTFa9fcxF7YXAuRxFvDVkv8LqeyyRR7XyhKftcsvbYmVyNm7FkYtLOF62SpZc5C+wUFJGuSph/QRMdhHCRYAiOAJKwD4aYnkvYsBAR18KYOEHI03GOCXKkTSiLUwYjdkDfHu0uUzagvfKMtdqlU3x6BSlN7JImpDxBWJ1m6niinRX7m/dYe6q7jejvpF5DYiX6xP6lm2b+V6dqkejiSNfgUU2RZlR1buqS6K6om5tfqpLkEBGncIfigrCrldM+m1oT69pVb5mOv+lMxaq9m+YmeFe3pAHbP8c5C2r7Jdsq2WcHxfJxOuostrGDPZrnIco4RQVV8r7BI57wbFwYt8adcf+ZamRSzRa+LePhA6qxl3c=</latexit><latexit sha1_base64="8rA1zy3wraoZ54fWw8n59vFsH9U=">AAAC13icjVHLSsNAFD2Nr1pftS7dBIvgqiQi6LLoxmUF+xBbyySdtqFpEiYTsZTiTtz6A271j8Q/0L/wzpiCWkQnJDlz7j1n5t7rRL4XS8t6zRhz8wuLS9nl3Mrq2vpGfrNQi8NEuLzqhn4oGg6Lue8FvCo96fNGJDgbOj6vO4MTFa9fcxF7YXAuRxFvDVkv8LqeyyRR7XyhKftcsvbYmVyNm7FkYtLOF62SpZc5C+wUFJGuSph/QRMdhHCRYAiOAJKwD4aYnkvYsBAR18KYOEHI03GOCXKkTSiLUwYjdkDfHu0uUzagvfKMtdqlU3x6BSlN7JImpDxBWJ1m6niinRX7m/dYe6q7jejvpF5DYiX6xP6lm2b+V6dqkejiSNfgUU2RZlR1buqS6K6om5tfqpLkEBGncIfigrCrldM+m1oT69pVb5mOv+lMxaq9m+YmeFe3pAHbP8c5C2r7Jdsq2WcHxfJxOuostrGDPZrnIco4RQVV8r7BI57wbFwYt8adcf+ZamRSzRa+LePhA6qxl3c=</latexit><latexit sha1_base64="8rA1zy3wraoZ54fWw8n59vFsH9U=">AAAC13icjVHLSsNAFD2Nr1pftS7dBIvgqiQi6LLoxmUF+xBbyySdtqFpEiYTsZTiTtz6A271j8Q/0L/wzpiCWkQnJDlz7j1n5t7rRL4XS8t6zRhz8wuLS9nl3Mrq2vpGfrNQi8NEuLzqhn4oGg6Lue8FvCo96fNGJDgbOj6vO4MTFa9fcxF7YXAuRxFvDVkv8LqeyyRR7XyhKftcsvbYmVyNm7FkYtLOF62SpZc5C+wUFJGuSph/QRMdhHCRYAiOAJKwD4aYnkvYsBAR18KYOEHI03GOCXKkTSiLUwYjdkDfHu0uUzagvfKMtdqlU3x6BSlN7JImpDxBWJ1m6niinRX7m/dYe6q7jejvpF5DYiX6xP6lm2b+V6dqkejiSNfgUU2RZlR1buqS6K6om5tfqpLkEBGncIfigrCrldM+m1oT69pVb5mOv+lMxaq9m+YmeFe3pAHbP8c5C2r7Jdsq2WcHxfJxOuostrGDPZrnIco4RQVV8r7BI57wbFwYt8adcf+ZamRSzRa+LePhA6qxl3c=</latexit>Published as a conference paper at ICLR 2022

Next, we interpret HyperDQN from an algorithmic level introduced in (Osband et al., 2018). In
particular, HyperDQN achieves the desiderata:

• Task-relevant feature. A task-specific feature can be obtained by the end-to-end training in (4.2).
Such a feature is good in the sense that it is optimized to approximate multiple value functions as
different invocations of z yield several individual TD loss functions. As discussed previously, this
feature learning procedure is beyond the scope of RLSVI.

• Commitment.2 The agent executes action sequences that span multiple periods by obeying its
intent. In each episode, HyperDQN samples a specific value function and takes greedy actions
according to this value function until the episode terminates. Unlike BootDQN (Osband et al.,
2016a) 3 and OFU-based algorithms (Ta¨ıga et al., 2020; Rashid et al., 2020; Bai et al., 2021),
HyperDQN does not combine ϵ-greedy as the perturbation in ϵ-greedy ruins “far-sighted” behaviors
generated by the original algorithm.

• Cheap computation. As discussed earlier, RLSVI needs to re-compute the feature covariance
matrix when the feature mapping changes, which results in a huge computation burden. To address
this issue, BootDQN simultaneously trains tens of ensembles. When the computation resources are
limited, the quality of generated posterior samples by BootDQN is poor (Lu & Van Roy, 2017).
However, our approach learns the approximate posterior samples in a meta way and the hypermodel
has certain generalization ability. As such, and our method tends to be more computation efficient.

As a side note, if the random index z is repeatedly sampled from a finite set, the hypermodel
degenerates to finite ensembles. As such, BootDQN can be viewed as a special case of HyperDQN.
Following (Osband et al., 2018), we summarize BootDQN and HyperDQN in Table 1.

Table 1: Important issues in posterior approximations for deep RL. A green tick indicates a satisfying
result, a red cross implies an undesirable result and a yellow circle means something in between.

BootDQN (Osband et al., 2018)
HyperDQN
RLSVI (Osband et al., 2016b)

Task-relevant feature
✓
✓
✗

5 EXPERIMENTS

Commitment Cheap computation

●
✓
✓

✗
●
✗✗

In this section, we present numerical experiments to validate the efficiency of the proposed method.
Experiment details can be found in Appendix C.

5.1 ATARI

Our first experiment is on the Arcade Learning Environment (Bellemare et al., 2013), which provides
a platform to assess the general competence. The training and evaluation procedures (e.g., observation
preprocessing and reward clipping) follows (Mnih et al., 2015) and (van Hasselt et al., 2016). We
measure the sample efficiency in terms of the human-normalized score during the interaction. In
particular, the raw score of each game is normalized so that 0% corresponds to a random agent and
100% to a human expert.
We consider five baselines: DQN (Mnih et al., 2015), OPIQ4 (Rashid et al., 2020), OB2I5 (Bai
et al., 2021), BootDQN6 (Osband et al., 2018) and NoisyNet7 (Fortunato et al., 2018). In particular,
OPIQ and OB2I are two advanced OFU (i.e, exploration bonus based) methods while BootDQN and
NoisyNet are randomized exploration methods.

The learning curve over 49 games (by median) with the 20M frames training budget is displayed
in Figure 3. We see that HyperDQN quickly explores and achieves the best performance among
baselines. The performance of OFU-based methods is bad: OPIQ cannot achieve a satisfying

2This concept is originally defined in the context of concurrent RL (Dimakopoulou & Roy, 2018). We borrow

this concept to help discuss the combination with ϵ-greedy under the standard RL framework.

3BootDQN uses ϵ-greedy for complicated tasks like Atari; see (Osband et al., 2016a, Appendix D.1).
4https://github.com/oxwhirl/opiq
5https://github.com/Baichenjia/OB2I
6Modified from https://github.com/johannah/bootstrap_dqn for a fair comparison.
7Modified from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master/

fqf_iqn_qrdqn/network.py for a fair comparison.

7

Published as a conference paper at ICLR 2022

Table 2: Comparison of algorithms on Atari in terms of the median over 49 games’ maximum
human-normalized scores. Note that the performance of DQN is based on 200M training frames
while other methods are based on 20M training frames.

DQN (200M) OPIQ OB2I BootDQN NoisyNet HyperDQN
110%
93%

37% 50%

82%

91%

performance over all games and OB2I degenerates after 10M frames. This is partially because the
exploration bonus may guide a direction that is unrelated to the environment reward. However,
randomized exploration methods do not have such an issue. Our experiment results regarding OFU-
based methods are consistent with (Ta¨ıga et al., 2020): though they may perform well on some hard
exploration tasks, current OFU-based methods cannot provide meaningful gains over the vanilla
DQN method on the whole Atari suite.

We notice that researchers also consider the metric of the maximum human-normalized score, which
measures the performance of the best policy during training; refer to (Mnih et al., 2015; van Hasselt
et al., 2016; Hessel et al., 2018; Fortunato et al., 2018). Specifically, we compute the maximum
evaluation scores after training for each game and average them by the median. We report such results
in Table 2. In particular, we see that HyperDQN outperforms the performance obtained by DQN with
200 training frames. We do not consider the statistics of mean because it is significantly affected by
some special games (e.g., Atlantis); see Table 5 in Appendix.

We note that the 20M frames training budget (rather than 200M frames) is commonly used in (Lee
et al., 2019; Bai et al., 2021). The main reason is that algorithms can achieve satisfactory performance
on many tasks with 20M frames while the 200M frames training budget requires about 30/10 days
with a CPU/GPU machine for a game, which is expansive in time and money.

Figure 3: Comparison of algorithms in terms of the
human-normalized score over 49 games in Atari.

Figure 4: Comparison of algorithms on Q*bert.
ϵ-greedy makes HyperDQN worse.

To better understand the performance of HyperDQN, we visualize the relative improvements over
baselines on “easy exploration” and “hard exploration” environments in Figure 12, 13, 14, and 15 in
Appendix D. We see that HyperDQN could perform well on many easy and hard exploration tasks.
However, HyperDQN does not work for Montezuma’s Revenge, in which the reward is very sparse so
there is limited feedback for feature selection. For the same reason, randomized exploration methods
in (Osband et al., 2018; Fortunato et al., 2018) do not perform well on this task.

Finally, we provide evidence that commitment is crucial for our method. Specifically, ϵ-greedy would
lead to worse performance for HyperDQN; see the empirical result on the game Q*bert in Figure 4.
The same observation holds for other environments (refer to Appendix D).

5.2 SUPERMARIOBROS

In this part, we evaluate algorithms on the SuperMarioBros suite (Kauten, 2018). Environment
preprocessing and algorithm parameters basically follow the one used in Atari and we do not tune
parameters for any algorithm.

We note that SuperMarioBros-1-3 and SuperMarioBros-2-2 are two hard exploration tasks due to
the long planning horizon and sparse reward. Experiments are run with 3 random seeds. Similar to
Table 5, we report the maximum scores in the following Table 3; see Figure 17 in Appendix for the
learning curves on each game. We see that HyperDQN beats baselines on 5 out of 9 games.

8

05101520Frame (millions)0%20%40%Median human-normalized scoreHyperDQNNoisyNetBootDQNOB2IOPIQDQN05101520Frame (millions)0%40%80%Median human-normalized scoreQ*bertHyperDQNHyperDQN(with epsilon-greedy)NoisyNetBootDQNOB2IOPIQDQNPublished as a conference paper at ICLR 2022

Table 3: Comparison of algorithms on SuperMarioBros in terms of the raw scores by the best policies
with 20M training frames.

SuperMarioBros-1-1
SuperMarioBros-1-2
SuperMarioBros-1-3
SuperMarioBros-2-1
SuperMarioBros-2-2
SuperMarioBros-2-3
SuperMarioBros-3-1
SuperMarioBros-3-2
SuperMarioBros-3-3

DQN
1, 070
2, 883
667
10, 800
813
3, 373
2, 560
11, 633
1, 007

OPIQ
7, 650
5, 515
2, 053
21, 654
1, 630
4, 718
3, 700
20, 872
2, 440

OB2I BootDQN NoisyNet HyperDQN
7, 924
4, 457
8, 267
4, 695
6, 047
1, 583
23, 047
14, 226
1, 984
1, 588
5, 980
4, 402
48, 385
3, 251
41, 140
26, 508
5, 568
3, 009

12, 439
6, 347
1, 587
14, 017
1, 808
6, 490
11, 310
33, 489
5, 886

7, 009
5, 665
1, 609
26, 415
1, 092
5, 108
3, 862
20, 955
2, 650

5.3 DEEP SEA
In this part, we consider the standard testbed for hard exploration: deep sea (Osband et al., 2018;
2020). In particular, there are two actions in this environment: move left and move right; see Figure 5.
The reward is sparse and is only released when the agent always takes the “right” action to obtain the
treasure at the corner. The maximum episode return is 0.99.

Following (Dwaracherla et al., 2020), we measure the computation complexity by

computation complexity = nsgd × nz × K,
(5.1)
where nsgd is the number of SGD steps per iteration, nz is the number of ensemble (index) samples,
and K is the minimum number of episode with return 0.99. This criterion is reasonable since without
the restriction of the computation complexity, methods may perfectly approximate RLSVI and enjoy
the same sample complexity.

The averaged empirical result with 10 random seeds is shown in Figure 6. In particular, the x-axis
corresponds to the size of the deep sea, i.e., the number of rows in Figure 5, and the y-axis corresponds
to the computation complexity as in (5.1). We see that HyperDQN is more computationally efficient
than BootDQN. Other methods fail to solve the deep sea when the size is larger than 20, which is
also observed in (Osband et al., 2018).

Figure 5: Illustration for deep sea.

Figure 6: Comparison of HyperDQN and BootDQN in
terms of computation complexity on the deep sea.

6 CONCLUSION AND FUTURE WORK

In this work, we present a practical exploration method to address the limitations of RLSVI and
BootDQN. To reinforce the central idea, we leverage the hypermodel (Dwaracherla et al., 2020) and
extend it from the bandit tasks to RL problems. Several algorithmic designs are developed to release
the power of randomized exploration.

Future directions include extending the idea for continuous control (Lillicrap et al., 2016; Haarnoja
et al., 2018) (i.e., building a randomized actor-critic with a similar architecture with HyperDQN; see
Appendix E.5), utilizing the developed epistemic uncertainty qualification tool for offline RL (Fuji-
moto et al., 2019; Levine et al., 2020), and acquiring an informative prior from human demonstrations
to accelerate exploration (Hester et al., 2018; Sun et al., 2018) (see Appendix E.6).

9

PublishedasaconferencepaperatICLR2020(a)Summaryscore(b)Examininglearningscaling.Figure2:Selectedoutputfrombsuiteevaluationon‘memorylength’.consequencesofitsactionstowardscumulativerewards,anagentseekingto‘explore’mustconsiderhowitsactionscanpositionittolearnmoreeﬀectivelyinfuturetimesteps.Theliteratureoneﬃcientexplorationbroadlystatesthatonlyagentsthatperformdeepexplo-rationcanexpectpolynomialsamplecomplexityinlearning(Kearns&Singh,2002).Thisliteraturehasfocused,forthemostpart,onuncoveringpossiblestrategiesfordeepexplo-rationthroughstudyingthetabularsettinganalytically(Jakschetal.,2010;Azaretal.,2017).Ourapproachinbsuiteistocomplementthisunderstandingthroughaseriesofbehaviouralexperimentsthathighlighttheneedforeﬃcientexploration.ThedeepseaproblemisimplementedasanN×Ngridwithaone-hotencodingforstate.Theagentbeginseachepisodeinthetopleftcornerofthegridanddescendsonerowpertimestep.EachepisodeterminatesafterNsteps,whentheagentreachesthebottomrow.IneachstatethereisarandombutﬁxedmappingbetweenactionsA={0,1}andthetransitions‘left’and‘right’.Ateachtimestepthereisasmallcostr=−0.01/Nofmovingright,andr=0formovingleft.However,shouldtheagenttransitionrightateverytimestepoftheepisodeitwillberewardedwithanadditionalrewardof+1.Thispresentsaparticularlychallengingexplorationproblemfortworeasons.First,followingthe‘gradient’ofsmallintermediaterewardsleadstheagentawayfromtheoptimalpolicy.Second,apolicythatexploreswithactionsuniformlyatrandomhasprobability2−Nofreachingtherewardingstateinanyepisode.ForthebsuiteexperimentweruntheagentonsizesN=10,12,..,50andlookattheaverageregretcomparedtooptimalafter10kepisodes.Thesummary‘score’computesthepercentageofrunsforwhichtheaverageregretdropsbelow0.9fasterthanthe2Nepisodesexpectedbydithering.Figure3:Deep-seaexploration:asimpleexamplewheredeepexplorationiscritical.DeepSeaisagoodbsuiteexperimentbecauseitistargeted,simple,challenging,scalableandfast.Byconstruction,anagentthatperformswellonthistaskhasmasteredsomekeypropertiesofdeepexploration.Oursummaryscoreprovidesa‘quickanddirty’waytocompareagentperformanceatahighlevel.OursweepoverdiﬀerentsizesNcanhelptopro-videempiricalevidenceofthescalingpropertiesofanalgorithmbeyondasimplepass/fail.Figure3presentsexampleoutputcomparingA2C,DQNandBootstrappedDQNonthis61015202530Deep Sea02004006008001000120014001600Computation Complexity (x K)AlgorithmHyperDQNBootDQNPublished as a conference paper at ICLR 2022

ACKNOWLEDGMENTS AND DISCLOSURE OF FUNDING

We thank Chenjia Bai for sharing some training results of OB2I, Hao Liang for the insightful
discussion, and Tian Xu for reading the manuscript and providing valuable comments. The work of
Z.-Q. Luo is supported by the National Natural Science Foundation of China (No. 61731018) and the
Guangdong Provincial Key Laboratory of Big Data Computation Theories and Methods.

ETHICS STATEMENT

This work focuses on designing an efficient exploration method to improve the sample efficiency
of reinforcement learning. This work may help reinforcement learning to be better used in the real
world. There could be some unexpected consequences if the reinforcement learning is abused. For
example, a robot is trained maliciously to hurt people.

REPRODUCIBILITY STATEMENT

First, a formal statement and proof of Theorem 1 are given in Appendix B.3. Second, the imple-
mentation details of the proposed algorithm HyperDQN can be found in Appendix A.2. Third, the
experiment details of numerical results in Section 5 are provided in Appendix C.

REFERENCES

Mohammad Gheshlaghi Azar, Ian Osband, and R´emi Munos. Minimax regret bounds for reinforce-
ment learning. In Proceedings of the 34th International Conference on Machine Learning, pp.
263–272, 2017.

Chenjia Bai, Lingxiao Wang, Lei Han, Jianye Hao, Animesh Garg, Peng Liu, and Zhaoran Wang.
Principled exploration via optimistic bootstrapping and backward induction. In Proceedings of the
38th International Conference on Machine Learning, pp. 577–587, 2021.

Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environ-
ment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:
253–279, 2013.

Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and R´emi Munos.
Unifying count-based exploration and intrinsic motivation. In Advances in Neural Information
Processing Systems 29, pp. 1471–1479, 2016.

Marc G. Bellemare, Will Dabney, and R´emi Munos. A distributional perspective on reinforcement
learning. In Proceedings of the 34th International Conference on Machine Learning, pp. 449–458,
2017.

Ronen I. Brafman and Moshe Tennenholtz. R-MAX - A general polynomial time algorithm for
near-optimal reinforcement learning. Journal of Maching Learning Research, 3:213–231, 2002.

Yuri Burda, Harrison Edwards, Deepak Pathak, Amos J. Storkey, Trevor Darrell, and Alexei A. Efros.
Large-scale study of curiosity-driven learning. In Proceedings of the 7th International Conference
on Learning Representations, 2019a.

Yuri Burda, Harrison Edwards, Amos J. Storkey, and Oleg Klimov. Exploration by random network
distillation. In Proceedings of the 7th International Conference on Learning Representations,
2019b.

Qi Cai, Zhuoran Yang, Chi Jin, and Zhaoran Wang. Provably efficient exploration in policy optimiza-
tion. In Proceedings of the 37th International Conference on Machine Learning, pp. 1283–1294,
2020.

Maria Dimakopoulou and Benjamin Van Roy. Coordinated exploration in concurrent reinforcement
learning. In Proceedings of the 35th International Conference on Machine Learning, pp. 1270–
1278, 2018.

10

Published as a conference paper at ICLR 2022

Vikranth Dwaracherla, Xiuyuan Lu, Morteza Ibrahimi, Ian Osband, Zheng Wen, and Benjamin
Van Roy. Hypermodels for exploration. In Proceedings of the 8th International Conference on
Learning Representations, 2020.

Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Matteo Hessel, Ian Osband,
Alex Graves, Volodymyr Mnih, R´emi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell,
In Proceedings of the 6th International
and Shane Legg. Noisy networks for exploration.
Conference on Learning Representations, 2018.

Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without
In Proceedings of the 36th International Conference on Machine Learning, pp.

exploration.
2052–2062, 2019.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy
maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the 35th
International Conference on Machine Learning, pp. 1856–1865, 2018.

Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney,
Dan Horgan, Bilal Piot, Mohammad Gheshlaghi Azar, and David Silver. Rainbow: Combining
improvements in deep reinforcement learning. In Proceedings of the 32nd AAAI Conference on
Artificial Intelligence, pp. 3215–3222, 2018.

Todd Hester, Matej Vecer´ık, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan,
John Quan, Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John P. Agapiou, Joel Z.
Leibo, and Audrunas Gruslys. Deep q-learning from demonstrations. In Proceedings of the 32nd
AAAI Conference on Artificial Intelligence, pp. 3223–3230, 2018.

Haque Ishfaq, Qiwen Cui, Viet Nguyen, Alex Ayoub, Zhuoran Yang, Zhaoran Wang, Doina Precup,
and Lin F. Yang. Randomized exploration in reinforcement learning with general value function
approximation. In Proceedings of the 38th International Conference on Machine Learning, pp.
4607–4616, 2021.

Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement

learning. Journal of Maching Learning Research, 11:1563–1600, 2010.

Chi Jin, Zeyuan Allen-Zhu, S´ebastien Bubeck, and Michael I. Jordan. Is q-learning provably efficient?

In Advances in Neural Information Processing Systems 30, pp. 4868–4878, 2018.

Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I. Jordan. Provably efficient reinforcement
learning with linear function approximation. In Proceedings of the 33rd Annual Conference on
Learning Theory, pp. 2137–2143, 2020.

Sham Machandranath Kakade. On the sample complexity of reinforcement learning. PhD thesis,

University of London London, England, 2003.

Christian Kauten. Super Mario Bros for OpenAI Gym. GitHub, 2018. URL https://github.

com/Kautenja/gym-super-mario-bros.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of

the 3rd International Conference on Learning Representations, 2015.

Yann A LeCun, L´eon Bottou, Genevieve B Orr, and Klaus-Robert M¨uller. Efficient backprop. In

Neural networks: Tricks of the trade, pp. 9–48. Springer, 1998.

Su Young Lee, Choi Sungik, and Sae-Young Chung. Sample-efficient deep reinforcement learning
via episodic backward update. In Advances in Neural Information Processing Systems 32, pp.
2110–2119, 2019.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial,

review, and perspectives on open problems. arXiv, 2005.01643, 2020.

Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa,
David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning.
In
Proceedings of the 4th International Conference on Learning Representations, 2016.

11

Published as a conference paper at ICLR 2022

Xiuyuan Lu and Benjamin Van Roy. Ensemble sampling.

In Advances in Neural Information

Processing Systems 30, pp. 3260–3268, 2017.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare,
Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control
through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Brendan O’Donoghue, Ian Osband, R´emi Munos, and Volodymyr Mnih. The uncertainty bellman
In Proceedings of the 35th International Conference on Machine

equation and exploration.
Learning, pp. 3836–3845, 2018.

Ian Osband. Risk versus uncertainty in deep learning: Bayes, bootstrap and the dangers of dropout.

In NIPS workshop on bayesian deep learning, 2016a.

Ian Osband. Deep Exploration via Randomized Value Functions. PhD thesis, Stanford University,

USA, 2016b.

Ian Osband and Benjamin Van Roy. Why is posterior sampling better than optimism for reinforcement
learning? In Proceedings of the 34th International Conference on Machine Learning, pp. 2701–
2710, 2017.

Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) efficient reinforcement learning via
posterior sampling. In Advances in Neural Information Processing Systems 26, pp. 3003–3011,
2013.

Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via
bootstrapped DQN. In Advances in Neural Information Processing Systems 29, pp. 4026–4034,
2016a.

Ian Osband, Benjamin Van Roy, and Zheng Wen. Generalization and exploration via randomized
value functions. In Proceedings of the 33rd International Conference on Machine Learning, pp.
2377–2386, 2016b.

Ian Osband, John Aslanides, and Albin Cassirer. Randomized prior functions for deep reinforcement

learning. In Advances in Neural Information Processing Systems 31, pp. 8626–8638, 2018.

Ian Osband, Benjamin Van Roy, Daniel J. Russo, and Zheng Wen. Deep exploration via randomized

value functions. Journal of Machine Learning Research, 20(124):1–62, 2019.

Ian Osband, Yotam Doron, Matteo Hessel, John Aslanides, Eren Sezener, Andre Saraiva, Katrina
McKinney, Tor Lattimore, Csaba Szepesv´ari, Satinder Singh, Benjamin Van Roy, Richard S. Sutton,
David Silver, and Hado van Hasselt. Behaviour suite for reinforcement learning. In Proceedings of
the 8th International Conference on Learning Representations, 2020.

Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven exploration
by self-supervised prediction. In Proceedings of the 34th International Conference on Machine
Learning, pp. 2778–2787, 2017.

John Quan and Georg Ostrovski. DQN Zoo: Reference implementations of DQN-based agents.

GitHub, 2020. URL http://github.com/deepmind/dqn_zoo.

Tabish Rashid, Bei Peng, Wendelin Boehmer, and Shimon Whiteson. Optimistic exploration even
with a pessimistic initialisation. In Proceedings of the 8th International Conference on Learning
Representations, 2020.

Daniel Russo. Worst-case regret bounds for exploration via randomized value functions. In Advances

in Neural Information Processing Systems 32, pp. 14410–14420, 2019.

Daniel Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. A tutorial on

thompson sampling. Foundations and Trends in Machine Learning, 11(1):1–96, 2018.

Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning
Research, 15(1):1929–1958, 2014.

12

Published as a conference paper at ICLR 2022

Bradly C. Stadie, Sergey Levine, and Pieter Abbeel. Incentivizing exploration in reinforcement

learning with deep predictive models. arXiv, 1507.00814, 2015.

Ruoyu Sun. Optimization for deep learning: theory and algorithms. arXiv, 1912.08957, 2019.

Wen Sun, J. Andrew Bagnell, and Byron Boots. Truncated horizon policy search: Combining
reinforcement learning & imitation learning. In Proceedings of the 6th International Conference
on Learning Representations, 2018.

Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction. MIT press, 2018.

Adrien Ali Ta¨ıga, William Fedus, Marlos C. Machado, Aaron C. Courville, and Marc G. Bellemare.
On bonus based exploration methods in the arcade learning environment. In Proceedings of the
8th International Conference on Learning Representations, 2020.

Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, Xi Chen, Yan Duan, John Schulman,
Filip De Turck, and Pieter Abbeel. #exploration: A study of count-based exploration for deep
reinforcement learning. In Advances in Neural Information Processing Systems 30, pp. 2753–2762,
2017.

Saran Tunyasuvunakool, Alistair Muldal, Yotam Doron, Siqi Liu, Steven Bohez, Josh Merel, Tom
Erez, Timothy Lillicrap, Nicolas Heess, and Yuval Tassa. dm control: Software and tasks for
continuous control. Software Impacts, 6:100022, 2020.

Hado van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-
learning. In Proceedings of the 30th AAAI Conference on Artificial Intelligence, pp. 2094–2100,
2016.

Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas.
In Proceedings of the 33rd

Dueling network architectures for deep reinforcement learning.
International Conference on Machine Learning, pp. 1995–2003, 2016.

Andrea Zanette, David Brandfonbrener, Emma Brunskill, Matteo Pirotta, and Alessandro Lazaric.
Frequentist regret bounds for randomized least-squares value iteration. In Proceedings of the 23rd
International Conference on Artificial Intelligence and Statistics, pp. 1954–1964, 2020.

13

Published as a conference paper at ICLR 2022

APPENDIX: HYPERDQN: A RANDOMIZED EXPLORATION
METHOD FOR DEEP REINFORCEMENT LEARNING

CONTENTS

1

Introduction

2 Background

3 Related Work

4 Methodology

4.1 Architecture Design .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

4.2 Differences with Dwaracherla et al. (2020): How the Direct Extension Fails .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

4.3 Training Objective .

.

.

4.4 Theoretical Guarantee .

4.5 The Algorithm .

.

.

.

.

5 Experiments

5.1 Atari .

.

.

.

.

.

.

5.2 SuperMarioBros .

5.3 Deep Sea .

.

.

.

.

.

.

.

.

.

.

.

.

.

6 Conclusion and Future Work

A Algorithm Details

A.1 Randomized Least-square Value Iteration (RLSVI) .

A.2 Implementation of HyperDQN .

.

A.3 Sampling From Unit Hypersphere .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

B Connection Between Bayesian Linear Regression and Hypermodel

B.1 Bayesian Linear Regression .

B.2 Hypermodel

.

.

.

.

.

B.3 Proof of Theorem 1 .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

B.4 Why Independent Gaussian Noise Cannot Work For Hypermodel? .

C Experiment Details

C.1 Algorithm Implementation And Parameters .

C.2 Environment Preprocessing .

C.3 Bayesian Linear Regression .

C.4 Parameter Choice .

.

.

.

.

.

D Additional Results

D.1 Atari .

.

.

.

.

.

.

D.2 SuperMarioBros .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

14

1

3

3

4

4

4

5

5

6

7

7

8

9

9

16

16

17

18

19

19

20

20

22

23

23

24

25

25

26

26

27

Published as a conference paper at ICLR 2022

E Discussion

E.1 Direct Extension of RLSVI Could Fail

E.2 Trainability Issues of HyperDQN .

.

.

.

.

E.3 ϵ-greedy in Many Efficient Algorithms .

E.4 Discussion of NoisyNet

.

.

.

.

.

E.5 Extension to Continuous Control

.

.

.

.

.

.

.

.

E.6 When an Informative Prior is Available .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

32

32

32

32

34

34

35

15

Published as a conference paper at ICLR 2022

A ALGORITHM DETAILS

A.1 RANDOMIZED LEAST-SQUARE VALUE ITERATION (RLSVI)

In this part, we offer a brief introduction to randomized least-square value iteration (RLSVI) (Osband
et al., 2016b; 2019). We hope this helps readers understand how RLSVI works and the underlying
limitations of RLSVI.

In its original form, RLSVI is designed for the episodic Markov Decision Process for easy analysis.
Here the word of “episodic” means the planning horizon (i.e., the episode length) is a finite number
T > 0. Given a feature map ϕ : S × A → Rd in advance, RLSVI assumes the optimal Q-value
function is a linear model with respect to this feature map. That is, for any stage t = 0, 1, · · · , T − 1,
RLSVI assumes

t ∈ Rd is the unknown optimal parameter and Q⋆

where θ⋆
Note that ϕt differs from stage to stage, but they are all fixed along the training.

Q⋆

t (s, a) = ϕt(s, a)⊤θ⋆
t ,

(A.1)
t is the optimal value function at stage t.

With the above assumption, RLSVI takes three steps to balance the trade-off between exploration and
exploitation.

• (Updating) Conditioned on the collected data, RLSVI estimates the optimal parameter θ⋆

t in a
Bayesian way. In particular, RLSVI obtains the posterior distribution over θ⋆
t via solving a Bayesian
linear regression problem8. More specifically, the Bayesian linear regression problem consists of
the fixed feature ϕt and the label yt generated by dynamic programming; see (A.3). Furthermore,
the posterior distribution can be obtained by the closed-form solution in (A.4).

• (Sampling) With the posterior distribution over θ⋆

t , RLSVI samples a specific parameter (cid:101)θt, which

forms the Q-value function (cid:101)Qt(s, a) = ϕt(s, a)⊤ (cid:101)θt.

• (Interaction) RLSVI takes the greedy action argmaxa (cid:101)Q(st, a) at stage t to collect new data.

Let k indicates the episode count and t indicates the stage/period count. Let [T ] indicates the set
{0, 1, 2, · · · , T − 1}. For instance, θ†
k,t ∈ Rd means the posterior mean at episode k and stage t.
With these notations, the procedure of RLSVI is outlined in Algorithm 1.

0,t ← prior mean θt, posterior covariance Σ†
0,t, Σ0,t) for t ∈ [T ].

for stage t = 0, 1, 2, · · · , T − 1 do

Algorithm 1 RLSVI (Osband et al., 2016b; 2019)
1: posterior mean θ†
2: sample (cid:101)θt ∼ N (θ†
3: for episode k = 0, 1, 2, · · · do
4:
5:
6:
7:
8:
9:
10:

end for
for stage t = T − 1, T − 2, · · · , 0 do

observe state st.
take the greedy action at = argmaxa ϕ(st, a)⊤ (cid:101)θt.
receive reward r(st, at).

θ†
k+1,t, Σ†
sample (cid:101)θt ∼ N (θ†

k,t, Σ†

k,t).

11:
12:
13: end for

end for

k+1,t ← update the posterior distribution (see (A.4)).

0,t ← prior covariance σ2

pI, ∀t ∈ [T ].

▷ Interaction

▷ Update

▷ Sampling

For Line 10 in Algorithm 1, RLSVI leverages the tool of Bayesian linear regression. Abstractly,
given the feature matrix X ∈ RN ×d (N is the number of samples) and the label vector Y ∈ RN , the
posterior distribution over θ⋆ can be computed in a closed-form way. More concretely, RLSVI forms

8Bayesian linear regression is reviewed in Appendix B.1.

16

Published as a conference paper at ICLR 2022

Xt and Yt with the state-action pairs collected at stage t up to episode k:

Xt =






ϕt(s0,t, a0,t)
ϕt(s1,t, a1,t)
· · ·
ϕt(sk,t, ak,t)


 ∈ R(k+1)×d,


Yt =


 ∈ Rk+1,







y0,t
y1,t
· · ·
yk,t

where

(cid:26)

yk,t =

rk,t + maxa(ϕ⊤

t+1 (cid:101)θt+1)(sk,t+1, a)
rk,t

if t < T − 1
if t = T − 1

(A.2)

(A.3)

With such defined Xt and Yt, RLSVI can compute the posterior distribution N (E[θ⋆
D]) over θ⋆
t :

t | D], Cov[θ⋆
t |

E[θ⋆

t | D] := θ†

k+1,t =

Cov[θ⋆

t | D] := Σ†

k+1,t =

(cid:18) 1
σ2
ω
(cid:18) 1
σ2
ω

X ⊤

t Xt +

X ⊤

t Xt +

(cid:19)−1 (cid:18) 1
σ2
ω

(cid:19)−1

.

1
σ2
p

1
σ2
p

I

I

X ⊤

t Yt +

(cid:19)

θt

,

1
σ2
p

(A.4)

Remark 2. As the feature is fixed in RLSVI, there exists an efficient implementation of RLSVI
based on the incremental update. Specifically, let Φ be the feature covariance matrix, i.e.,

Φk+1,t =

1
σ2
ω

X ⊤

t Xt +

1
σ2
p

I.

For easy presentation, let ϕk,t denote ϕ(sk,t, ak,t). Then, we have the incremental update formula:

Φk+1,t = Φk,t +

1
σ2
ω

ϕk,tϕ⊤

k,t, with Φ0,t =

1
σ2
p

I.

Furthermore, we have a more efficient update rule for the posterior covariance in (A.4) by the
Sherman–Morrison formula (see Lemma 1). More concretely, we have that
(1/σ2

k+1,t := Σ†
Σ†

k,t −

ω) · Σ†
1 + (1/σ2

k,tϕk,tϕ⊤
k,tΣ†
ω) · ϕ⊤

k,tΣ†
k,t
k,tϕk,t

, with Σ†

0,t = σ2

pI.

t Yt in (A.4) can also be implemented in an incremental update way. In

As a side note, the term X ⊤
short, the posterior distribution is easy to compute when the feature is fixed.
Lemma 1 (Sherman–Morrison formula). Suppose A ∈ Rn×n is an invertible square matrix and
u, v ∈ Rn are column vectors. Then A + uv⊤ is invertible if and only if 1 + v⊤A−1u ̸= 0. In this
case,

(cid:0)A + uv⊤(cid:1)−1

= A−1 −

A−1uv⊤A−1
1 + v⊤A−1u

.

Here, uv⊤ is the outer product of two vectors u and v.

A.2

IMPLEMENTATION OF HYPERDQN

In this part, we provide more implementation details of HyperDQN.

Recall that we work with two value functions: the prior one and the differential one. The prior value
function and the differential value function are independently constructed, meaning there are no
shared parameters. The actual Q-value is the weighted sum of two terms:

Q(s, a, z) = βprior · Qθprior,fνprior (z)(s, a) + βdifferential · Qθhidden,fν (z)(s, a),
where βprior > 0 and βdifferential > 0 are two positive scalars. When βprior = βdifferential = 1, we recover
the formulation in (4.2).

(A.5)

The feature extractor (i.e., the hidden layers) are task-specific, which will be discussed in Appendix C.
For the hypermodel, the differential part is initialized based on the default initializer of PyTorch
while the initialization of the prior hypermodel is a little tricky. Following (Dwaracherla et al., 2020),
each row of νw is sampled from the unit hypersphere and νb is based on the mean of the desired
prior distribution. The purpose of this initialization is to guarantee the θprior = fνprior (z) follows a
desired Gaussian distribution. For example, if we want to obtain a prior distribution N (1, I) for

17

Published as a conference paper at ICLR 2022

Algorithm 2 HyperDQN

generate a random vector z ∼ N (0, I).
instantiate the Q-value function Qθ(s, a, z).
for stage t = 0, 1, 2, · · · , T − 1 do

observe state st.
take the greedy action a ← argmaxa Qθ(st, a, z).
receive the next state st+1 and reward r(st, at).
sample ξ uniformly from unit hypersphere.
store (st, at, r, ξ, st+1) into the replay buffer D.
agent step n ← n + 1.
if mod (agent step n, train frequency M ) == 0 then

1: agent step n ← 0, train step ℓ ← 0.
2: for episode k = 0, 1, 2, · · · do
3:
4:
5:
6:
7:
8:
9:
10:
11:
12:
13:
14:
15:
16:
17:
18:
19:
20:
21:
22: end for

end for

end if

update the target network.

sample a mini-batch (cid:101)D of (s, a, r, ξ, s′) from the replay buffer D.
sample N random vectors z′
i: (cid:101)Z = {z′
optimize ν and θhidden using the empirical loss function (A.6) with (cid:101)D and (cid:101)Z.
train step ℓ ← ℓ + 1.

i=1.

i}N

end if
if mod (train step ℓ, target update frequency G) == 0 then

▷ Sampling

▷ Interaction

▷ Update

θprior := fνprior(z), it suffices to set νb = 1 and to uniformly sample νw from the unit hypersphere,
where 1 is a vector with elements of 1.

According to the population objective function in (4.2),
L(ν, θhidden; (cid:101)Z, (cid:101)D) given the set (cid:101)Z and mini-batch (cid:101)D is

the empirical objective function

|D|
| (cid:101)D|

(cid:88)

(cid:18)

(s,a,r,ξ,s′)∈ (cid:101)D

Qtarget(s′, z) + σωz⊤ξ − Qprediction(s, a, z)

(cid:19)2



∥fν(z)∥2

 ,

+

σ2
ω
σ2
p

(A.6)



(cid:88)



z∈ (cid:101)Z

1
| (cid:101)Z|

where

Qprediction(s, a, z) = βprior · Qθprior,fνprior (z)(s, a) + βdifferential · Qθhidden,fν (z)(s, a),

Qtarget(s′, z) = r + γ max
a′

(cid:105)
(cid:104)
βprior · Qθprior,fνprior (z)(s′, a′) + βdifferential · Q¯θhidden,f¯ν (z)(s′, a′)

.

In practice, (A.6) can be efficiently solved by Adam (Kingma & Ba, 2015). In particular, we sample
multiple z and the mini-batch from the experience replay buffer to construct the empirical objective
function. As such, Adam can be applied to optimize the hidden layers and the hypermodel as
illustrated in Figure 1.
Remark 3 (Feature Representation for Prior Value Functions). Different from supervised learning,
the prior value function is crucial for online decision making (Osband et al., 2018). For the prior value
functions, we not only need to consider the prior hypermodel but also the prior feature representation.
Importantly, the feature extractor for the prior value function should be fixed rather than being shared
with the differential part. In other words, we use two independent feature networks (i.e., the hidden
layers) for the prior value function and the differential value function, respectively. Instead, the output
of the prior value function would change if the prior value function and the differential value function
share the same hidden layers, which violates the intuition that the prior value function should not be
affected by the learning process.

A.3 SAMPLING FROM UNIT HYPERSPHERE

For completeness, we describe the method for uniform sampling over unit hypersphere. There are
two steps:

18

Published as a conference paper at ICLR 2022

(1) Generate x = (x1, x2, . . . , xd), using a zero mean and unit variance Gaussian distribution.

Thus, the probability density of x
1

p(x) =

(2π)

d
2

(cid:18)

exp

−

x2
1 + x2

2 + · · · + x2
d

(cid:19)

2

is spherically symmetric.

(2) Normalize the vector x = (x1, x2, . . . , xd) to a unit vector, namely x/∥x∥, which gives a
sample uniformly over the unit hypersphere. Note that once the vector is normalized, its
coordinates are no longer statistically independent.

B CONNECTION BETWEEN BAYESIAN LINEAR REGRESSION AND

HYPERMODEL

B.1 BAYESIAN LINEAR REGRESSION

In this part, we review the Bayesian linear regression. Let x ∈ Rd be the feature and y = ⟨θ⋆, x⟩ + ω⋆
ω) is the observation noise. Here, we treat θ⋆ as a random variable
be the label, where ω⋆ ∈ N (0, σ2
and pose a prior distribution p0 : N (θp, σ2
pI) on θ⋆. Given the dataset D = {(xi, yi)}N
i=1, we can
update the posterior probability density over θ⋆ by the Bayes rule:

p(θ⋆ | D) ∝ p (cid:0){(xi, yi)}N

i=1 | θ⋆(cid:1) · p0(θ⋆).

Thanks to the conjugate prior, the posterior distribution over θ⋆ is also a Gaussian distribution
N (E[θ⋆ | D], Cov[θ⋆ | D]) with

E[θ⋆ | D] =

Cov[θ⋆ | D] =

(cid:18) 1
σ2
ω
(cid:18) 1
σ2
ω

X ⊤X +

X ⊤X +

(cid:19)−1 (cid:18) 1
σ2
ω

(cid:19)−1

,

1
σ2
p

1
σ2
p

I

I

X ⊤Y +

(cid:19)

θp

,

1
σ2
p

(B.1)

(B.2)

where

X =






x⊤
1
x⊤
2
· · ·
x⊤
N


 ∈ RN ×d,


Y =


 ∈ RN .







y1
y2
· · ·
yN

In summary, given the collected dataset (i.e., the features and labels), we can obtain posterior samples
through sampling from N (E[θ⋆ | D], Cov[θ⋆ | D]).

Recently, Osband et al. (2019) provide another way of obtaining posterior samples through optimizing
a randomized least square problem.
Lemma 2 ((Osband et al., 2019)). Let x be the feature vector and the target y be generated by
y = ⟨θ⋆, x⟩ + ω⋆, where ω⋆ ∼ N (0, σ2
ω) is the noise and the prior on θ⋆ is N (θp, σ2
pI). Let
ω be the algorithmic noise that follows the same distribution with ω⋆ (i.e., ω ∼ N (0, σ2
ω)) and
(cid:98)θ ∼ N (θp, σ2

pI) be a prior sample. Then,

or

argmin
θ

(cid:88)

(y + ω − θ⊤x)2 +

(x,y,ω)∈D

σ2
ω
σ2
p

(cid:13)
(cid:13)
2
(cid:13)
(cid:13)
(cid:13)θ − (cid:98)θ
(cid:13)
2

,

(cid:98)θ + argmin

θ

(cid:88)

(x,y,ω)∈D

(y + ω − (θ + (cid:98)θ)⊤x)2 +

σ2
ω
σ2
p

∥θ∥2
2 ,

(B.3)

(B.4)

could yield a sample from the posterior distribution N (E[θ⋆ | D], Cov[θ⋆ | D]) defined by (B.1) and
(B.2).

First, it is obvious that (B.3) and (B.4) are equivalent by the variable change trick. Let us focus on
(B.4) to explain the main idea of Lemma 2. Notice that the posterior sample includes two parts: the
prior (cid:98)θ and the differential θ (θ is a specific optimal solution to (B.4)). In particular, θ is a random
variable, which depends on the randomness in ω. Based on this observation, we can leverage the

19

Published as a conference paper at ICLR 2022

optimality condition to obtain the optimal solution of (B.4). Subsequently, we can verify that the
mean of (cid:98)θ + θ and its covariance are identical with the posterior mean E[θ⋆ | D] and posterior
covariance Cov[θ⋆ | D], respectively.
Remark 4. Importantly, Lemma 2 recasts a sampling problem to an optimization problem, which
provides a purely computational cue to obtain a posterior sample. Consequently, it provides another
way of implementing RLSVI by solving the least square problem in each iteration. However, there is
no clear advantage for this implementation under the case where RLSVI can work (e.g., tabular or
linear MDPs).

B.2 HYPERMODEL

In this part, we provide more explanation about the hypermodel. To proceed, note that the main
issue of the procedure in Lemma 2 is its computational efficiency. The reason is that we have to
solve multiple randomized least square problems if we want to obtain many posterior samples. One
of the motivations of the hypermodel is to address this computational issue. Indeed, the original
optimization problem in (Dwaracherla et al., 2020) is
(cid:0)y + σωz⊤ξ − gfν (z)(x)(cid:1)2

(cid:13)fν(z) − fνprior (z)(x)(cid:13)
(cid:13)
2
(cid:13)

(cid:20) (cid:88)

(dz),

(B.5)

p(z)

+

(cid:90)

(cid:21)

min
ν

z

(x,y,ξ)∈D

σ2
ω
σ2
p

where νprior = ν0 is the fixed prior hypermodel parameter. Consider the linear case where gfν (z)(x) =
fν(z)⊤x, then (B.5) becomes
(cid:90)





(cid:88)

(cid:0)y + σωz⊤ξ − fν(z)⊤x(cid:1)2

+

∥fν(z) − fν0(z)∥2

 (dz),

min
ν

z

p(z)



(x,y,ξ)∈D

σ2
ω
σ2
p

which is very similar to (B.3) and the difference is discussed in Remark 1. Intuitively, the hypermodel
wants to solve “infinite” randomized least-square problems simultaneously. Through this, the
hypermodel can obtain multiple posterior samples by sampling z.

Note that (B.5) has a different form with the loss function in (4.1). However, as discussed in
(Dwaracherla et al., 2020, Section 2.5), these two loss functions are equivalent. To derive (4.1)
from (B.5), we take three steps: 1) replace fν(z) in (B.5) with fνprior (z) + fν′(z); 2) let gfν (z)(x) =
gfνprior (z)+fν′ (z)(x) = gfνprior (z)(x) + gfν′ (z)(x); 3) change the optimization variable from ν to ν′.

B.3 PROOF OF THEOREM 1

Here, we consider the linear hypermodel, i.e., fν(z) = ν⊤
By the property of Gaussian distribution, we know that fν(z) ∼ N (νb, ν⊤
Therefore, νb and ν⊤
By the formulation, there are two models in a complete hypermodel: the prior one and the differential
one. Specifically,

w z + νb, where νw ∈ RNz×d and νb ∈ Rd.
w νw) as z ∼ N (0, I).

w νw qualify the mean and covariance, respectively.

• The prior hypermodel is

f(cid:98)ν(z) = (cid:98)ν⊤

w z + (cid:98)νb,

where (cid:98)ν = ((cid:98)νw, (cid:98)νb) is fixed and a non-trainable variable. Here, we set the (cid:98)ν⊤
w = σpI and
(cid:98)νb = θp as the correct prior parameters, then f(cid:98)ν(z) is a sample from the prior distribution
N (θp, σ2

pI).

• The differential hypermodel is

where ν = (νw, νb) is a trainable variable.

fν(z) = ν⊤

w z + νb,

Let the sample (xi, yi) generated under Assumption 1. We augment each data sample (xi, yi) ∈ D
with ξi independently sampled from unit hypersphere (refer to Appendix A.3). Then, the dataset
becomes

D = {(xi, yi, ξi) : i = 1, 2, . . . , |D|} ,

20

Published as a conference paper at ICLR 2022

with the augmented noise realizations

Ξ = {ξi : i = 1, . . . , |D|}.

w, ν⋆

Theorem 1 (Formal statement). Under Assumption 1, set νprior
ν⋆ = (ν⋆
θ := fνprior (z) + fν⋆ (z) ∼ N (νprior

= θp. Let
b ) be the optimal solution of (4.1) conditioned on specific realizations of ξ , then
w)⊤(νprior
w + ν⋆

w) = Cov[θ⋆ | D] + err(Ξ),

b , (νprior
w + ν⋆

b + ν⋆
(νprior

w = σpI and νprior

b = E[θ⋆ | D],

w)⊤(νprior

w + ν⋆

w + ν⋆

νprior
b + ν⋆

w)) with

b

where

err(Ξ) := Cov[θ⋆ | D]





1
σ2
ω

(cid:88)

xξ⊤ξ′x′⊤ +

(x,ξ)̸=(x′,ξ′)∈D

1
σωσp

(cid:88)

(x,ξ)∈D

(xξ⊤ + ξx⊤)


 Cov[θ⋆ | D]

Furthermore, the error term satisfies EΞ[err(Ξ)] = 0.

Proof of Theorem 1. For simplicity, we omit the subscript ⋆ for the optimal solution to (4.1) in the
proof. That is, we use the shorthand notation ν = (νω, νb) for the optimal solution to (4.1).

The objective function (4.1) for learning hypermodel given the data D becomes:
(cid:0)y + σωz⊤ξ − x⊤((νw + (cid:98)νw)⊤z + (νb + (cid:98)νb))(cid:1)2

(cid:18) (cid:88)

L(ν; D) =

p(z)

(cid:90)

z

(x,y,ξ)∈D
According to the first-order optimality condition:

(cid:13)
(cid:13)ν⊤

w z + νb

2 (cid:19)
(cid:13)
(cid:13)

(dz)

+

σ2
ω
σ2
p

∂L
∂νb

= Ez





(cid:88)

(x,y,ξ)∈D

(−x) (cid:0)y + σωξ⊤z − x⊤((νw + (cid:98)νw)⊤z + (νb + (cid:98)νb))(cid:1) +

σ2
ω
σ2
p

(νw

⊤z + νb) | D





(B.6)

(cid:88)

=

(x,y,ξ)∈D

x(x⊤νb + x⊤

(cid:98)νb − y) +

σ2
ω
σ2
p

νb = 0.

It is straightforward to obtain that

(cid:88)

(x,y,ξ)∈D

x(x⊤νb + x⊤

(cid:98)νb − y) +

σ2
ω
σ2
p

(νb + (cid:98)νb) =

σ2
ω
σ2
p

(cid:98)νb.

Then, we can infer that

νb + (cid:98)νb =

xx⊤ +

(cid:32)

x∈D

(cid:88)

1
σ2
ω
(cid:18) 1
σ2
ω
= E [θ⋆ | D] ,

=

X ⊤X +

1
σ2
p

I

(cid:33)−1 


1
σ2
p
(cid:19)−1 (cid:18) 1
σ2
ω

I

1
σ2
ω

(cid:88)

xy +

(x,y)∈D



(cid:98)νb



1
σ2
p

X ⊤Y +

(cid:19)

1
σ2
p

θp

where X ∈ R|D|×d, Y ∈ R|D|, and E [θ | D] is defined in (B.1). This implies that the hypermodel
can recover the posterior mean.
For the variable νw, we calculate its partial derivative ∂L/∂ν⊤

w as



(ν⊤

w z + νb)z⊤ | D



σ2
ω
σ2
p


w zz⊤ | D
ν⊤

 .

(B.7)



(cid:88)

Ez



(x,y,ξ)∈D

(cid:0)y + σωz⊤ξ − x⊤((νw + (cid:98)νw)⊤z + (νb + (cid:98)νb))(cid:1) (−xz⊤) +





= Ez

(cid:88)

(x,y,ξ)∈D

(cid:0)−σωz⊤ξxz⊤ + x⊤(νw + (cid:98)νw)⊤zxz⊤(cid:1) +

σ2
ω
σ2
p

To proceed, we utilize the following helpful lemma.
Lemma 3. Let z ∼ N (0, Id). For any fixed vector a ∈ Rd, we have

Ez

(cid:2)a⊤zxz⊤(cid:3) = xa⊤.

21

Published as a conference paper at ICLR 2022

Proof. The proof is easy if we look at each entry of the matrix,

(cid:2)Ez

(cid:2)z⊤axz⊤(cid:3)(cid:3)

ij = Ez

(cid:2)z⊤a[xz⊤]ij

(cid:3) = Ez

(cid:33)

(cid:35)

akzk

xizj

= ajxi = [xa⊤]ij.

(cid:34)(cid:32) d
(cid:88)

k=1

Then, with the above Lemma 3 and (B.7), we further calculate

∂L
∂ν⊤
w

=

(cid:88)

(cid:0)−σωxξ⊤ + xx⊤(νw + (cid:98)νw)(cid:1) +

(x,y,ξ)∈D

σ2
ω
σ2
p

ν⊤
w .

(B.8)

By the first order optimality condition, we have

(νw + (cid:98)νw) =

(cid:18) 1
σ2
ω

X ⊤X +

(cid:19)−1

1
σ2
p

I





1
σ2
ω

(cid:88)

σωxξ⊤ +

(x,ξ)∈D



(cid:98)νw

 .

1
σ2
p

With the posterior covariance in (B.2)

Σ−1 := Cov (θ⋆ | D) =

(cid:18) 1
σ2
ω

X ⊤X +

(cid:19)−1

,

1
σ2
p

I

and define the set Ξ = (ξ1, . . . , ξ|D|) corresponding to the ξi in (xi, yi, ξi) ∈ D, we obtain that



(νw + (cid:98)νw)⊤(νw + (cid:98)νw)
1
σ2
ω

= Σ−1

(cid:88)



(cid:88)

(x,ξ)

(x′,ξ′)

xξ⊤ξ′x′⊤ +

1
σ4
p

(cid:98)ν⊤
w (cid:98)νw +

1
ωσ2
σ2
p

(cid:88)

(x,ξ)

(cid:0)σωxξ⊤

(cid:98)νw + σω (cid:98)ν⊤

w ξx⊤(cid:1)


 Σ−1

= Σ−1





1
σ2
ω

X ⊤X +

1
σ2
p

I +

1
σ2
ω

(cid:88)

xξ⊤ξ′x′⊤ +

(x,ξ)̸=(x′,ξ′)

1
σωσp

(cid:88)

(xξ⊤ + ξx⊤)


 Σ−1

(x,ξ)

= Σ−1ΣΣ−1 + err(Ξ),

(cid:80)

where the second equality is due to the fact that ∥ξ∥ = 1 and we set (cid:98)νw = σpI at the beginning;
and err(Ξ) = Σ−1 (cid:16) 1
Σ−1. Taking the
expectation over Ξ = (ξ1, . . . , ξ|D|), we have


 = 0, EΞ

(x,ξ)̸=(x′,ξ′) xξ⊤ξ′x′⊤ + 1
σωσp

(cid:17)
(x,ξ)(xξ⊤ + ξx⊤)

(xξ⊤ + ξ⊤x)

xξ⊤ξ′x′⊤

 = 0.

EΞ

(cid:88)

(cid:88)

(cid:80)

σ2
ω









(x,ξ)̸=(x′,ξ′)

(x,ξ)

Finally, we have

(cid:2)(νw + (cid:98)νw)⊤(νw + (cid:98)νw)(cid:3) = Σ−1ΣΣ−1 + EΞ [err(Ξ)] = Σ−1.
This implies the hypermodel can also recover the posterior covariance in expectation.

EΞ

B.4 WHY INDEPENDENT GAUSSIAN NOISE CANNOT WORK FOR HYPERMODEL?

Following similar steps in Appendix B.3, we want to argue that the posterior approximation result
cannot be achieved if we replace z⊤ξ with an independent Gaussian noise ω in (4.1).
Similarly, we augment each sample (xi, yi) ∈ D with ωi ∼ N (0, σ2
(ω1, ω2, . . . , ω|D|)⊤ as the noise vector. The dataset becomes

ω) and denote ω =

D = (xi, yi, ωi : i = 1, 2, . . . , |D|).

Now, the objective function for learning hypermodel given the data D becomes:

Lω(ν; D) =

(cid:90)

z

(cid:18) (cid:88)

p(z)

(x,y,ω)∈D

(cid:0)y + ω − x⊤((νw + (cid:98)νw)⊤z + (νb + (cid:98)νb))(cid:1)2

+

(cid:13)
(cid:13)ν⊤

w z + νb

2 (cid:19)
(cid:13)
(cid:13)

(dz).

σ2
ω
σ2
p

22

Published as a conference paper at ICLR 2022

Similar to (B.6), we have the first order optimality condition

∂Lω
∂νb

=

(cid:88)

(x,y,ω)

x(x⊤b + x⊤

(cid:98)νb − y + ω) +

σ2
ω
σ2
p

νb = 0.

Then, we have

1
σ2
p
For the variable νw, similar to (B.7), we calculate the partial derivative ∂L/∂ν⊤

X ⊤(Y + ω) +

νb + (cid:98)νb =

X ⊤X +

1
σ2
p

θp

I

(cid:19)−1 (cid:18) 1
σ2
ω

(cid:18) 1
σ2
ω

(cid:19)

w as

̸= E [θ | D] .



(cid:88)

Ez



(x,y,ξ)∈D

(cid:0)y + ω − x⊤((νw + (cid:98)νw)⊤z + (νb + (cid:98)νb))(cid:1) (−xz⊤) +





= Ez

(cid:88)

(x,y,ω)∈D

(cid:0)−ωxz⊤ + x⊤(νw + (cid:98)νw)⊤zxz⊤(cid:1) +

σ2
ω
σ2
p

w zz⊤ | D
ν⊤



σ2
ω
σ2
p




(ν⊤

w z + νb)z⊤ | D







= Ez

(cid:88)

(cid:0)x⊤(νw + (cid:98)νw)⊤zxz⊤(cid:1) +

(x,y,ω)∈D



w zz⊤ | D
ν⊤



σ2
ω
σ2
p

(cid:88)

=

(x,y,ω)∈D

(cid:0)xx⊤(νw + (cid:98)νw)⊤(cid:1) +

σ2
ω
σ2
p

ν⊤
w .

By the first order optimality condition, we have

(νw + (cid:98)νw)⊤ =

(cid:18) 1
σ2
ω

X ⊤X +

1
σ2
p

(cid:19)−1 (cid:18) 1
σ2
p

I

(cid:19)

.

(cid:98)ν⊤

w

Therefore, by the fact that (cid:98)ν⊤

w (cid:98)νw = σ2

pI,

(νw + (cid:98)νw)⊤(νw + (cid:98)νw) = Σ−1

(cid:18) 1
σ2
p
This implies that an independent Gaussian noise ω cannot work and the z-dependent noise is
indispensable for posterior approximation.

Σ−2 ̸= Cov (θ | D) .

Σ−1 =

1
σ2
p

(cid:19)

I

C EXPERIMENT DETAILS

C.1 ALGORITHM IMPLEMENTATION AND PARAMETERS

Common Hyperparameters. All agents use the same network structure as in (Mnih et al., 2015) on
Atari and SuperMarioBros:

state → conv(32, 8, 4) → relu → conv(64, 4, 2) → relu → conv(64, 3, 1)
→ mlp(512) → relu → mlp(number of actions),

where conv(32, 8, 4) means a convolution layer with 64 filters of size 8 and stride of 4, and mlp(512)
means a fully-connected layer with output size of 512, and relu stands for Rectified Linear Units.

The algorithmic parameters on Atari and SuperMarioBros basically follow (Mnih et al., 2015, Table
1). For example, the replay buffer size is 1M; the batch size is 32; the discount factor is 0.99; the
target network update frequency is 10K agent steps; the train frequency is 4 agent steps; and the
replay starts after 50K agent steps. For algorithms with ϵ-greedy (e.g., DQN, BootDQN, OPIQ and
OB2I), the exploration ϵ is annealed from 1.0 to 0.1 linearly (from 50K agent steps to 1M agent steps,
respectively); the test ϵ is 0.05.

DQN. The training result of DQN (Mnih et al., 2015) on Atari is based on DQN Zoo (Quan & Ostro-
vski, 2020)9. We implement DQN with tianshou10 framework and train it on SuperMarioBros.

9https://github.com/deepmind/dqn_zoo/blob/master/results.tar.gz
10https://github.com/thu-ml/tianshou

23

Published as a conference paper at ICLR 2022

OPIQ. We use the implementation in the public repository https://github.com/oxwhirl/
opiq. For a fair comparison, we do no use the mixed Monte Carlo return and set the final ϵ to be
0.05. Other parameters follow (Rashid et al., 2020).

OB2I. We use the implementation for OB2I in the public repository https://github.com/
Baichenjia/OB2I. The training data of OB2I on Atari is partially shared by authors in (Bai et al.,
2021) (private communication). All parameters follow (Bai et al., 2021).

BootDQN. We modify the implementation for BootDQN from the public repository https://
github.com/johannah/bootstrap_dqn to make a fair comparison. In particular, BootDQN
uses 10 independent ensembles and the ϵ-greedy strategy is same with DQN. We use the version with
prior value functions (Osband et al., 2018). Note that we do not implement the “vote” mode because
we observe that it does not matter in practice.

NoisyNet. We modify the implementation for NoisyNet
https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master/fqf_
iqn_qrdqn/network.py to make a fair comparison.
In particular, NoisyNet re-samples a
noisy network for action selection and update. The noise scale is 0.5 and other parameters follow
(Fortunato et al., 2018).

from the public repository

HyperDQN. We use the implementation as described in Appendix A.2. Algorithm parameters are
listed in Table 4. To stabilize training, each z corresponds to 32 mini-batch samples and the effective
batch size of HyperDQN is 32 × 10 = 320.

Table 4: Algorithmic parameters of HyperDQN for Atari and SuperMarioBros.

z dimension Nz
prior scale βprior in (A.5)
differential scale βdifferential in (A.5)
number of z for training in (A.6)
noise scale σw in (A.6)
prior regularization scale σp in (A.6)
learning rate

Atari
32
0.1
0.1
10
0.01
0.1
0.0001

SuperMarioBros Deep Sea
32
0.1
0.1
10
0.01
0.1
0.0001

2
10.0
1.0
20
0.0
0.0
0.001

C.2 ENVIRONMENT PREPROCESSING

Atari. We follow the same preprocessing as in (Mnih et al., 2015) and (van Hasselt et al., 2016).
In particular, there are random no-operation steps (up to 30) before the interaction. Each agent
step corresponds to 4 environment steps by repeating the same action while each environment step
corresponds to 4 frames of the simulator. The raw score is clipped to {−1, 0, +1} for training but the
evaluation performance is based on the raw score. Episodes are early stopped after 108K frames as
in (van Hasselt et al., 2016). The observation for the agent is based on 4 stacked frames, which is
reshaped to (4, 84, 84).

SuperMarioBros. The observation and action preprocessing are the same with Atari 2600 suite.
Two things are different: 1) there is no random “no operation”; 2) the training reward is based on
0.01×raw score rather than clipping. Moreover, the maximum episode length is 1500×4 = 6000
frames.

For algorithms (DQN, BootDQN, NoisyNet, HyperDQN) that we implement by the tianshou
framework, the training frequency is 10 and the target update frequency is 500 to accelerate training
speed. Other parameters are identical to the one for Atari.

Deep sea. The implementation of the deep sea task is from bsuite11 (Osband et al., 2020). The
neural network architecture is :

state → mlp(64) → relu → mlp(64) → relu → mlp(number of actions).

11https://github.com/deepmind/bsuite/blob/master/bsuite/environments/

deep_sea.py

24

Published as a conference paper at ICLR 2022

To make a fair comparison with BootDQN, we use the setting in (Osband et al., 2018). In particular,
the training frequency is 1, target update frequency is 4, batch size is 128, and buffer size is 200K.
For HyperDQN, its parameters are listed in Table 4. For BootDQN, it uses 10 ensembles and the
other parameters are identical to HyperDQN. Both algorithms do not use ϵ-greedy.

C.3 BAYESIAN LINEAR REGRESSION

The 2-dimensional Bayesian linear regression problem used in Section 4.3 is based on y = θw ·
x + θb + ϵ, where x, y, θw, θb, ϵ ∈ R. Specifically, ϵ is sampled from the Gaussian distribution. To
generate the dataset, x is sampled from [−4, 4] uniformly. The total number of training samples is 50.
The prior distribution for (θw, θb) is N (0, I2) while the actual value (θw, θb) is (6.06, 0.47) for the
dataset generation.

We obtain the exact posterior distribution by the Bayes update rule. For hypermodel, the dimension of
z is 2. We use gradient descent with a learning rate of 0.005 and momentum of 0.9. The number of of
the gradient descent iterations is 10000. After the optimization, we visualize the 5000 samples from
different distributions (e.g., the posterior distribution and the one transformed by the hypermodel) in
Figure 2. Indeed, the KL-divergence between the exact posterior distribution and the counterparts by
hypermodel(z⊤ξ) and hypermodel(ω) are 0.1 and 326.4, respectively.

C.4 PARAMETER CHOICE

Noise scale σω. In this part, we provide the ablation study about the noise scale σω. The numerical
result is displayed in Figure 7. We observe the performance of HyperDQN is not very sensitive to σω.

Figure 7: Comparison of HyperDQN with
different noise scales σω.

Figure 8: Comparison of HyperDQN with
different prior scales σp.

Prior scale σp. In this part, we provide the ablation study about the prior scale σp. The numerical
result is shown in Figure 8. We find that a large prior scale results in a poor performance. The reason
is that the posterior update is slow under this case.

Hypermodel architecture. In this paper, we mainly focus on the linear hypermodel. Here, we
investigate the variant with a non-linear hypermodel (i.e., an MLP hypermodel). In particular, we
consider the hypermodel is a two-layer neural network of width 64 with ReLU activation. We call this
variant as HyperDQN(MLP) and the original HyperDQN as HyperDQN(Linear). It is intuitive
that HyperDQN(MLP) has a more powerful posterior approximation ability but it may be hard to
train HyperDQN(MLP) since the architecture is more complex. The empirical results on Atari and
SuperMarioBros are shown in Figure 9. We see that HyperDQN does not obtain significant gains by
an MLP hypermodel. We conjecture the reason is the trainability issue of the MLP hypermodel.

Number of ensembles in BootDQN. In this paper, we implement BootDQN with 10 ensembles,
which follows the configuration in (Osband et al., 2016a, Section 6.1). However, in (Osband et al.,
2018), BootDQN is implemented with 20 ensembles. We remark that the choice of 10 ensembles
is commonly used in the previous literature (Rashid et al., 2020; Bai et al., 2021) since it is more
computationally cheap. For completeness, we provide the ablation study about this parameter choice;
see Figure 10 for the result.

25

05101520Frame (millions)0200040006000800010000Episode ScoreSuperMarioBros-1-2HyperDQN(=0.1)HyperDQN(=0.01)HyperDQN(=0.001)05101520Frame (millions)0200040006000800010000Episode ScoreSuperMarioBros-1-2HyperDQN(p=0.0)HyperDQN(p=0.1)HyperDQN(p=10.0)Published as a conference paper at ICLR 2022

(a) Episode return on Pong.

(b) Episode return on SuperMarioBros-1-2.

Figure 9: Comparison of HyperDQN with a linear hypermodel and a MLP hypermodel.

(a) Episode return on Pong.

(b) Episode return on SuperMarioBros-1-2.

Figure 10: Comparison of BootDQN with 10 ensembles and 20 ensembles.

D ADDITIONAL RESULTS

D.1 ATARI

For all algorithms, learning curves on each game are visualized in Figure 11. The maximum raw
scores on each individual game are reported in Table 5.

proposed−baseline

Relative improvement on each game. To better understand the improvement of HyperDQN, we
visualize the relative score compared with baselines. Specifically, the relative score is calculated as
max(human,baseline)−human (Wang et al., 2016). According to the taxonomy in (Bellemare et al., 2016),
we cluster environments by 4 groups: “hard exploration (dense reward)”, “hard exploration (sparse
reward)”, “easy exploration”, and “unknown”. See the results in Figure 12, Figure 13, Figure 14, and
Figure 15.

We observe that HyperDQN has improvements on both “easy exploration” environments (e.g., Battle
Zone, Jamesbond, and Pong) and “hard exploration” environments (e.g., Frostbite, Gravitar, and
Zaxxon). However, we notice that HyperDQN does not work well on very sparse reward tasks
like Montezuma’s Revenge. We explain the failure reason as follows. As we have discussed in the
introduction, without a good feature, randomized exploration methods are rarely competent. On
Montezuma’s Revenge, the extremely sparse reward provides limited feedback for feature selection.
As a result, it is expected that randomized exploration methods (including BootDQN and NoisyNet)
do not work well for this task. In contrast, prediction error based methods (Pathak et al., 2017; Burda
et al., 2019b; Rashid et al., 2020) could leverage specific architecture designs to provide auxiliary
reward feedback to help feature selection and exploration. As a result, these methods perform well
on Montezuma’s Revenge. We kindly remind that these methods do not perform well on other tasks
because specifically designed architectures in these methods do not generalize well as pointed out in
(Ta¨ıga et al., 2020).

Commitment of randomized exploration. Here we provide the evidence that using the ϵ-greedy
strategy contradicts the commitment and leads to poor performance for HyperDQN; see the results in
Figure 16. We observe that combing HyperDQN with ϵ-greedy results in poor performance.

26

05101520Frame (millions)3020100102030Episode ScorePongHyperDQN(Linear)HyperDQN(MLP)05101520Frame (millions)0200040006000800010000Episode ScoreSuperMarioBros-1-2HyperDQN(Linear)HyperDQN(MLP)05101520Frame (millions)21.020.820.620.420.220.019.8Episode ScorePongBootDQN(10 ensembles)BootDQN(20 ensembles)05101520Frame (millions)0100020003000400050006000Episode ScoreSuperMarioBros-1-2BootDQN(10 ensembles)BootDQN(20 ensembles)Published as a conference paper at ICLR 2022

Figure 11: Learning curves of algorithms on Atari. Solid lines correspond to the median performance
over 3 random seeds while shaded ares correspond to 90% confidence interval. Same with other
figures for Atari.

D.2 SUPERMARIOBROS

For all algorithms, learning curves on each game are visualized in Figure 17.

27

025050075010001250Alien060120180240300Amidar010002000300040005000Assault06001200180024003000Asterix030060090012001500Asteroids0.00.51.01.52.02.51e6Atlantis020040060080010001200Bank Heist02500500075001000012500Battlezone0100020003000400050006000Beam Rider01020304050Bowling603003060Boxing080160240320400Breakout0200040006000800010000Centipede020040060080010001200Chopper Command020000400006000080000100000120000Crazy Climber01500300045006000Demon Attack2520151050Double Dunk02004006008001000Enduro100806040200Fishing Derby0612182430Freeway050010001500200025003000Frostbite0100020003000400050006000Gopher060120180240300Gravitar0250050007500100001250015000H.E.R.O.2520151050Ice Hockey0100200300400500600James Bond 007010002000300040005000Kangaroo0200040006000800010000Krull0500010000150002000025000Kung-Fu Master0.000.010.020.030.040.05Montezumas Revenge05001000150020002500Ms. Pac-Man01500300045006000Name This Game3020100102030Pong8004000400Private Eye02500500075001000012500Q*bert015003000450060007500River Raid0800016000240003200040000Road Runner048121620Robotank05001000150020002500Seaquest0150300450600750Space Invaders0200040006000800010000Stargunner2520151050Tennis010002000300040005000Time Pilot04080120160200Tutankham05101520Frame (millions)020004000600080001000012000Upn Down05101520Frame (millions)080160240320400Venture05101520Frame (millions)0250005000075000100000125000Video Pinball05101520Frame (millions)0400800120016002000Wizard of Wor05101520Frame (millions)010002000300040005000ZaxxonHyperDQNNoisyNetBootDQNOB2IOPIQDQNPublished as a conference paper at ICLR 2022

Table 5: The maximal score over 200 evaluation episodes for the best policy in hindsight (af-
ter 20M frames) for Atari games. The performance of the random policy and the human
expert is from https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/
atari_data.py#L41-L101. The performance of OB2I is from (Bai et al., 2021).

Alien
Amidar
Assault
Asterix
Asteroids
Atlantis
Bank Heist
Battle Zone
BeamRider
Bowling
Boxing
Breakout
Centipede
Chopper Command
Crazy Climber
Demon Attack
Double Dunk
Enduro
Fishing Derby
Freeway
Frostbite
Gopher
Gravitar
H.E.R.O
Ice Hockey
Jamesbond
Kangaroo
Krull
Kung-Fu Master
Montezuma’s Revenge
Ms. Pacman
Name This Game
Pong
Private Eye
Q*Bert
River Raid
Road Runner
Robotank
Seaquest
Space Invaders
Star Gunner
Tennis
Time Pilot
Tutankham
Up and Down
Venture
Video Pinball
Wizard of Wor
Zaxxon

Random
227.8
5.8
222.4
210.0
719.1
12,850.0
14.2
2,360.0
363.9
23.1
0.1
1.7
2,090.9
811.0
10,780.5
152.1
-18.6
0.0
-91.7
0.0
65.2
257.6
173.0
1,027.0
-11.2
29.0
52.0
1,598.0
258.5
0.0
307.3
2,292.3
-20.7
24.9
163.9
1,338.5
11.5
2.2
68.4
148.0
664.0
-23.8
3,568.0
11.4
533.4
0.0
16,256.9
563.5
32.5

Human
7,127.7
1,719.5
742.0
8,503.3
47,388.7
29,028.1
753.1
13,454.5
16,926.5
160.7
12.1
30.5
12,017.0
7,387.8
35,829.4
1,971.0
-16.4
860.5
5.5
29.6
4,334.7
2,412.5
3,351.4
30,826.4
0.9
302.8
3,035.0
2,665.5
22,736.3
4,753.3
6,951.6
4,076.0
14.6
69,571.3
13,455.0
17,118.0
7,845.0
11.9
42,054.7
1,668.7
10,250.0
-8.3
5,229.2
167.6
11,693.2
1,187.5
17,667.9
4,756.5
9,173.3

OB2I
916.9
94.0
2,996.2
2,719.0
959.9
3,146,300.0
378.6
8,756.5
3,736.7
30.0
75.1
423.1
2,661.8
1,100.3
53,346.7
6,794.6
-18.2
719.0
-60.1
32.1
1,277.3
6,359.5
393.6
3,302.5
-4.2
434.3
2,387.0
45,388.8
16,272.2
0.0
1,794.9
8,576.8
18.7
1,174.1
4,275.0
2,926.5
21,831.4
13.5
332.1
904.9
1,290.2
-1.0
3,404.5
297.0
5,100.8
16.1
80,607.0
480.7
2,842.0

OPIQ BootDQN NoisyNet HyperDQN
2,910.0
565.7
2,083.0
2,283.3
3,190.0
880,233.3
470.0
24,333.3
3,955.3
64.0
56.3
43.7
9,923.0
2,733.3
117,966.7
8,755.0
2.7
456.0
-26.3
32.7
3,943.3
4,600.0
1,316.7
20,156.7
-1.7
700.0
10,166.7
8,413.3
26,933.3
0.0
4,590.0
7,106.7
21.0
2,810.0
16,616.7
8,020.0
27,033.3
19.7
1,360.0
925.0
3,633.3
5.3
10,166.7
213.3
16,520.0
966.7
172,896.3
7,266.7
10,066.7

2,623.3
319.0
3,182.7
3,466.7
3,353.3
835,166.7
270.0
22,666.7
7,002.0
90.3
60.7
212.0
11,051.7
1,900.0
88,400.0
9,148.3
-6.0
626.3
-54.3
9.3
1,853.3
5,466.7
966.7
13,590.0
-4.0
650.0
8,666.7
10,643.3
1,500.0
0.0
3,793.3
8,200.0
-17.7
7,641.0
5,250.0
8,003.3
17,266.7
11.7
2,180.0
1,390.0
13,300.0
-1.0
8,400.0
269.3
16,333.3
266.7
262,718.0
4,266.7
8,400.0

2,596.7
395.7
5,000.3
4,000.0
3,143.3
857,100.0
250.0
20,333.3
5,467.3
116.0
53.7
352.0
9,492.3
2,966.7
138,133.3
8,845.0
3.3
1,169.7
-3.7
25.0
4,020.0
7,606.7
1,250.0
12,675.0
0.7
650.0
2,333.3
10,260.0
34,933.3
0.0
4,440.0
9,093.3
2.5
11,018.7
5,966.7
5,993.3
3,533.3
37.7
230.0
1,410.0
22,066.7
-1.0
8,433.3
181.3
52,476.7
700.0
881,999.3
6,066.7
4,233.3

2,316.7
161.6
3,385.0
1,500.0
976.7
1,780,266.7
430.0
18,333.3
4,385.3
56.3
96.0
247.0
11,891.7
1,666.7
71,566.7
3,805.0
-18.7
1,033.3
-91.3
26.3
1,640.0
2,266.7
450.0
8,345.0
-11.7
350.0
2,400.0
3,763.3
18,033.3
0.0
2,186.7
4,883.3
-20.0
5,124.3
4,558.3
4,536.7
20,866.7
15.3
1,846.7
853.3
933.3
-20.3
4,100.0
184.0
37,893.3
133.3
53,924.7
2,500.0
3,300.0

Commitment of randomized exploration. Here we provide the evidence that using the ϵ-greedy
strategy contradicts the commitment and leads to poor performance for HyperDQN on SuperMario-
Bros; see the result in Figure 18.

In addition, we provide the evidence that the randomized exploration approach BootDQN (Osband
et al., 2016a) could obtain some gains if ϵ-greedy is not used; see the result in Figure 19. The result
suggests that the original implementation of BootDQN does not satisfy the commitment property.
Note that the variant without ϵ-greedy is still inferior compared with HyperDQN.

28

Published as a conference paper at ICLR 2022

Figure 12: Relative improvement of HyperDQN compared with OPIQ (Rashid et al., 2020) on Atari.
max(human,baseline)−human (Wang et al., 2016). Environments
The relative performance is calculated as
are grouped according to the taxonomy in (Bellemare et al., 2016, Table 1). “Unknown” indicates
such environments are not considered in (Bellemare et al., 2016).

proposed−baseline

Figure 13: Relative improvement of HyperDQN compared with OB2I (Bai et al., 2021) on Atari. The
proposed−baseline
relative performance is calculated as
max(human,baseline)−human (Wang et al., 2016). Environments are
grouped according to the taxonomy in (Bellemare et al., 2016, Table 1). “Unknown” indicates such
environments are not considered in (Bellemare et al., 2016).

29

BreakoutUp and DownEnduroAtlantisBoxingAssaultCentipedePrivate EyeBeamRiderSeaquestMontezuma's RevengeSpace InvadersAsteroidsBank HeistBowlingAlienAsterixChopper CommandTutankhamFreewayRiver RaidAmidarGravitarStar GunnerRoad RunnerRobotankMs. PacmanBattle ZoneKung-Fu MasterH.E.R.OFrostbiteFishing DerbyVentureZaxxonCrazy ClimberIce HockeyName This GameQ*BertGopherJamesbondWizard of WorPongDemon AttackTennisKrullKangarooVideo PinballTime PilotDouble Dunk-100%0%200%400%600%800%1000%Relative Score-83-57-56-51-41-41-20-3-3-10555699161722222427283034363840405467707476838691108109114116136165215260316365973Relative Score of HyperDQN Compared with OPIQHard Exploration (Dense Reward)Hard Exploration (Sparse Reward)UnknownEasy ExplorationBreakoutKrullAtlantisAssaultEnduroTutankhamGopherBoxingName This GameAsterixMontezuma's RevengeBeamRiderSpace InvadersFreewayPrivate EyeSeaquestAsteroidsPongBank HeistIce HockeyRoad RunnerStar GunnerBowlingChopper CommandAmidarTennisAlienGravitarDemon AttackRiver RaidFishing DerbyMs. PacmanKung-Fu MasterRobotankH.E.R.OFrostbiteJamesbondCentipedeZaxxonVentureQ*BertUp and DownBattle ZoneVideo PinballCrazy ClimberWizard of WorKangarooTime PilotDouble Dunk-100%0%200%400%600%800%1000%Relative Score-90-84-72-33-31-29-29-25-23-5011222561221242425252828292930323542475557626673798093102140143152162261407950Relative Score of HyperDQN Compared with OB2IHard Exploration (Dense Reward)Hard Exploration (Sparse Reward)UnknownEasy ExplorationPublished as a conference paper at ICLR 2022

Figure 14: Relative improvement of HyperDQN compared with BootDQN (Osband et al., 2018)
on Atari. The relative performance is calculated as
max(human,baseline)−human (Wang et al., 2016).
Environments are grouped according to the taxonomy in (Bellemare et al., 2016, Table 1). “Unknown”
indicates such environments are not considered in (Bellemare et al., 2016).

proposed−baseline

Figure 15: Relative improvement of HyperDQN compared with NoisyNet (Fortunato et al., 2018)
on Atari. The relative performance is calculated as
max(human,baseline)−human (Wang et al., 2016).
Environments are grouped according to the taxonomy in (Bellemare et al., 2016, Table 1). “Unknown”
indicates such environments are not considered in (Bellemare et al., 2016).

proposed−baseline

Figure 16: Comparison of HyperDQN with and without ϵ-greedy on Atari.

30

BreakoutStar GunnerAssaultVideo PinballSpace InvadersKrullTutankhamEnduroBowlingName This GameBeamRiderGopherAsterixCentipedeBoxingPrivate EyeDemon AttackSeaquestAsteroidsMontezuma's RevengeRiver RaidUp and DownAlienAtlantisJamesbondBattle ZoneGravitarMs. PacmanChopper CommandAmidarKangarooZaxxonIce HockeyH.E.R.OBank HeistTennisFishing DerbyTime PilotCrazy ClimberFrostbiteRoad RunnerVentureDouble DunkWizard of WorFreewayRobotankQ*BertPongKung-Fu Master-100%-50%0%50%100%150%Relative Score-80-77-37-36-31-25-22-20-19-19-18-17-14-11-7-7-4-2-00014588111213141718192227282937384957596972798286110113Relative Score of HyperDQN Compared with BootDQNUnknownHard Exploration (Sparse Reward)Easy ExplorationHard Exploration (Dense Reward)BreakoutStar GunnerVideo PinballUp and DownAssaultEnduroRobotankGopherBowlingSpace InvadersName This GameFishing DerbyKung-Fu MasterKrullAsterixIce HockeyCrazy ClimberPrivate EyeBeamRiderChopper CommandDouble DunkFrostbiteDemon AttackMontezuma's RevengeAsteroidsGravitarMs. PacmanSeaquestAtlantisCentipedeAlienBoxingJamesbondAmidarRiver RaidTutankhamWizard of WorBattle ZoneVentureH.E.R.OFreewayTennisBank HeistTime PilotPongZaxxonQ*BertKangarooRoad Runner-100%0%100%200%300%400%Relative Score-88-86-82-69-61-61-51-41-38-32-29-23-23-21-21-20-16-12-9-4-3-2-100223345581013192222222526283036526480263300Relative Score of HyperDQN Compared with NoisyNetUnknownHard Exploration (Sparse Reward)Easy ExplorationHard Exploration (Dense Reward)05101520Frame (millions)0612182430Episode ScoreFreewayHyperDQNHyperDQN(with epsilon-greedy)05101520Frame (millions)3020100102030PongHyperDQNHyperDQN(with epsilon-greedy)05101520Frame (millions)02500500075001000012500Q*bertHyperDQNHyperDQN(with epsilon-greedy)Published as a conference paper at ICLR 2022

Figure 17: Learning curves of algorithms on SuperMarioBros. Solid lines correspond to the median
performance over 3 random seeds while shaded ares correspond to 90% confidence interval. Same
with other figures for SuperMarioBros.

Figure 18: Comparison of HyperDQN with and without ϵ-greedy on SuperMarioBros.

Figure 19: Comparison of BootDQN with and without ϵ-greedy on SuperMarioBros.

31

03000600090001200015000Episode ScoreSuperMarioBros-1-10200040006000800010000SuperMarioBros-1-201500300045006000SuperMarioBros-1-30600012000180002400030000Episode ScoreSuperMarioBros-2-105001000150020002500SuperMarioBros-2-201500300045006000SuperMarioBros-2-305101520Frame (millions)01500030000450006000075000Episode ScoreSuperMarioBros-3-105101520Frame (millions)01000020000300004000050000SuperMarioBros-3-205101520Frame (millions)01500300045006000SuperMarioBros-3-3HyperDQNNoisyNetBootDQNOB2IOPIQDQN05101520Frame (millions)02500500075001000012500Episode ScoreSuperMarioBros-1-1HyperDQNHyperDQN(with epsilon-greedy)05101520Frame (millions)0200040006000800010000SuperMarioBros-1-2HyperDQNHyperDQN(with epsilon-greedy)05101520Frame (millions)01500300045006000SuperMarioBros-1-3HyperDQNHyperDQN(with epsilon-greedy)05101520Frame (millions)020004000600080001000012000Episode ScoreSuperMarioBros-1-1BootDQNBootDQN(without epsilon-greedy)05101520Frame (millions)015003000450060007500SuperMarioBros-1-2BootDQNBootDQN(without epsilon-greedy)05101520Frame (millions)010002000300040005000SuperMarioBros-1-3BootDQNBootDQN(without epsilon-greedy)Published as a conference paper at ICLR 2022

E DISCUSSION

E.1 DIRECT EXTENSION OF RLSVI COULD FAIL

In this part, we provide numerical evidence that the direct extension of RLSVI could fail for Atari
tasks. In particular, we use a randomly initialized convolutional neural network as the fixed feature
extractor and implement RLSVI according to Appendix A.1. Since we work with infinite horizon
MDPs, we replace the non-discounted target in (A.3) by the corresponding discounted target. We
display experiment results in Figure 20. In particular, we observe that such an implementation of
RLSVI is unable to solve Atari tasks.

Figure 20: Comparison of RLSVI and HyperDQN on Atari.

E.2 TRAINABILITY ISSUES OF HYPERDQN

In this part, we continue the discussion of the architecture design in Section 4.1. In particular, we
illustrate why the original architecture in (Dwaracherla et al., 2020) is not suitable for RL tasks.

Table 6: Information about the base model used for deep sea.

Expected Magnitude of Initialization (1/

fin)

√

Layer 1
Layer 2
Layer 3

Shape (fin, fout)
(900, 64)
(64, 64)
(64, 2)

1/30
1/8
1/8

As we have argued in Section 4.1, the main challenge of applying the architecture in (Dwaracherla
et al., 2020) is parameter initialization. To illustrate this issue, consider that we use a two-layer
MLP neural network with a width of 64 for the deep sea task of size 30. Suppose we use the default
PyTorch initialization12 (i.e., sampling from the uniform distribution U [−1/(cid:112)di−1, 1/(cid:112)di−1]),
which is also known as “1/
fin” with fin being the input dimension of the i-th layer. We list the
parameter information in Table 6 (recall that the state dimension is 900 and the action dimension is 2
for the deep sea task). We see that the expected magnitude of the parameter differs over layers. In
particular, the desired magnitude is quite small compared with the magnitude of the output of the
hypermodel (≈ 1). As a result, if we directly apply the hypermodel for all layers of the base model,
the input signal over layers explodes and the gradient is quite large.

√

• After about 10 iterations, we observe that the parameter diverges if we use the SGD algorithm

and apply the hypermodel for all layers of the base model.

• Even if we use adaptive algorithms like Adam, the training result is not expected; see the
empirical result in Figure 21. In particular, if we apply the hypermodel for all layers of the
base model, Q-values are very large and the resulting algorithm does not succeed.

Furthermore, this parameter initialization issue becomes more severe when we use deep convolution
neural networks for Atari and SuperMarioBros tasks.

E.3

ϵ-GREEDY IN MANY EFFICIENT ALGORITHMS

We realize many efficient algorithms (Osband et al., 2016a; Rashid et al., 2020; Bai et al., 2021)
still use ϵ-greedy in practice. However, we argue that ϵ-greedy is not efficient since it cannot write

12https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

32

05101520Frame (millions)08162432Episode ScoreFreewayHyperDQNRLSVI05101520Frame (millions)301501530PongHyperDQNRLSVI05101520Frame (millions)030006000900012000Q*bertHyperDQNRLSVIPublished as a conference paper at ICLR 2022

(a) Episode return on the deep sea.

(b) Q-values on the deep sea.

Figure 21: Comparison of HyperDQN with two configurations: 1) the hypermodel is applied only at
the last layer of the base model (ours); 2) the hypermodel is applied for all layers of the base model.

off sub-optimal actions after experimentation. In addition, combining with ϵ-greedy violates the
principle of commitment as discussed in Section 4.5. We conjecture existing practical algorithms
combine their exploration strategies with ϵ-greedy mainly due to imperfect imitation of theoretical
algorithms, which is analyzed below. This direction deserves further investigation.

On the one hand, we believe BootDQN (Osband et al., 2016a) uses ϵ-greedy is to improve the
diversity since finite ensembles could offer limited diverse action sequences. For instance, if the
number of actions is larger than the number of ensembles, BootDQN cannot produce all possible
action sequences to explore at the initial stage. In this case, ϵ-greedy is somehow helpful. Another
explanation is that BootDQN just follows the default setting of DQN (Mnih et al., 2015). In Figure 19,
we see that the variant that drops the ϵ-greedy could be better than the original implementation of
BootDQN in some tasks.

On the other hand, there are many practical issues to implement OFU-based algorithms in real
applications, as discussed in Section 3. In fact, directly adding an exploration bonus yields unexpected
outcomes. Let us (re-)illustrate this using the example of the deep sea (shown in Figure 5). Imagine
the agent visits a certain path to collect the data; for instance, the red path in Figure 22. With the
experience replay buffer, the agent can only update the Q-value function with state-action pairs from
the visited path. By exploration bonus, such state-action sequences would have larger Q-values.
During the next episode, the agent would repeat the same action sequences by following the greedy
policy. It turns out that theoretical algorithms like (Jin et al., 2018) do not have such an issue. This is
because theoretically, algorithms in (Jin et al., 2018) can use an optimistic initialization. With the
optimistic initialization, the algorithm instead would select other actions rather than visited actions.
In practice, however, it is not easy to obtain an optimistic initialization with deep neural networks
(Rashid et al., 2020). Again, ϵ-greedy could somehow avoid this issue.

Figure 22: Illustration for the initialization issue of OFU-based algorithms. After the initialization,
action “left” dominates at state s0. After the experience replay buffer, action “left” dominates again at
state s0. This issue is caused by the pessimistic initialization by neural network (Rashid et al., 2020).

In addition to the initialization issue, the uncertainty (i.e., the exploration bonus) should propagate
over stages in a backward manner.
In (Jin et al., 2018), this operation is implemented by the
exact dynamic programming. In contrast, practical algorithms randomly draw mini-batch from the

33

015000300004500060000Number of Interactions0.000.250.500.751.00Episode Return Deep Sea-30HyperDQN(only last layer)HyperDQN(all layers)015000300004500060000Number of Interactions0100101Q ValuesDeep Sea-30HyperDQN(only last layer)HyperDQN(all layers)PublishedasaconferencepaperatICLR2020(a)Summaryscore(b)Examininglearningscaling.Figure2:Selectedoutputfrombsuiteevaluationon‘memorylength’.consequencesofitsactionstowardscumulativerewards,anagentseekingto‘explore’mustconsiderhowitsactionscanpositionittolearnmoree ectivelyinfuturetimesteps.Theliteratureone cientexplorationbroadlystatesthatonlyagentsthatperformdeepexplo-rationcanexpectpolynomialsamplecomplexityinlearning(Kearns&Singh,2002).Thisliteraturehasfocused,forthemostpart,onuncoveringpossiblestrategiesfordeepexplo-rationthroughstudyingthetabularsettinganalytically(Jakschetal.,2010;Azaretal.,2017).Ourapproachinbsuiteistocomplementthisunderstandingthroughaseriesofbehaviouralexperimentsthathighlighttheneedfore cientexploration.ThedeepseaproblemisimplementedasanN◊Ngridwithaone-hotencodingforstate.Theagentbeginseachepisodeinthetopleftcornerofthegridanddescendsonerowpertimestep.EachepisodeterminatesafterNsteps,whentheagentreachesthebottomrow.IneachstatethereisarandombutﬁxedmappingbetweenactionsA={0,1}andthetransitions‘left’and‘right’.Ateachtimestepthereisasmallcostr=≠0.01/Nofmovingright,andr=0formovingleft.However,shouldtheagenttransitionrightateverytimestepoftheepisodeitwillberewardedwithanadditionalrewardof+1.Thispresentsaparticularlychallengingexplorationproblemfortworeasons.First,followingthe‘gradient’ofsmallintermediaterewardsleadstheagentawayfromtheoptimalpolicy.Second,apolicythatexploreswithactionsuniformlyatrandomhasprobability2≠Nofreachingtherewardingstateinanyepisode.ForthebsuiteexperimentweruntheagentonsizesN=10,12,..,50andlookattheaverageregretcomparedtooptimalafter10kepisodes.Thesummary‘score’computesthepercentageofrunsforwhichtheaverageregretdropsbelow0.9fasterthanthe2Nepisodesexpectedbydithering.Figure3:Deep-seaexploration:asimpleexamplewheredeepexplorationiscritical.DeepSeaisagoodbsuiteexperimentbecauseitistargeted,simple,challenging,scalableandfast.Byconstruction,anagentthatperformswellonthistaskhasmasteredsomekeypropertiesofdeepexploration.Oursummaryscoreprovidesa‘quickanddirty’waytocompareagentperformanceatahighlevel.Oursweepoverdi erentsizesNcanhelptopro-videempiricalevidenceofthescalingpropertiesofanalgorithmbeyondasimplepass/fail.Figure3presentsexampleoutputcomparingA2C,DQNandBootstrappedDQNonthis6PublishedasaconferencepaperatICLR2020(a)Summaryscore(b)Examininglearningscaling.Figure2:Selectedoutputfrombsuiteevaluationon‘memorylength’.consequencesofitsactionstowardscumulativerewards,anagentseekingto‘explore’mustconsiderhowitsactionscanpositionittolearnmoree ectivelyinfuturetimesteps.Theliteratureone cientexplorationbroadlystatesthatonlyagentsthatperformdeepexplo-rationcanexpectpolynomialsamplecomplexityinlearning(Kearns&Singh,2002).Thisliteraturehasfocused,forthemostpart,onuncoveringpossiblestrategiesfordeepexplo-rationthroughstudyingthetabularsettinganalytically(Jakschetal.,2010;Azaretal.,2017).Ourapproachinbsuiteistocomplementthisunderstandingthroughaseriesofbehaviouralexperimentsthathighlighttheneedfore cientexploration.ThedeepseaproblemisimplementedasanN◊Ngridwithaone-hotencodingforstate.Theagentbeginseachepisodeinthetopleftcornerofthegridanddescendsonerowpertimestep.EachepisodeterminatesafterNsteps,whentheagentreachesthebottomrow.IneachstatethereisarandombutﬁxedmappingbetweenactionsA={0,1}andthetransitions‘left’and‘right’.Ateachtimestepthereisasmallcostr=≠0.01/Nofmovingright,andr=0formovingleft.However,shouldtheagenttransitionrightateverytimestepoftheepisodeitwillberewardedwithanadditionalrewardof+1.Thispresentsaparticularlychallengingexplorationproblemfortworeasons.First,followingthe‘gradient’ofsmallintermediaterewardsleadstheagentawayfromtheoptimalpolicy.Second,apolicythatexploreswithactionsuniformlyatrandomhasprobability2≠Nofreachingtherewardingstateinanyepisode.ForthebsuiteexperimentweruntheagentonsizesN=10,12,..,50andlookattheaverageregretcomparedtooptimalafter10kepisodes.Thesummary‘score’computesthepercentageofrunsforwhichtheaverageregretdropsbelow0.9fasterthanthe2Nepisodesexpectedbydithering.Figure3:Deep-seaexploration:asimpleexamplewheredeepexplorationiscritical.DeepSeaisagoodbsuiteexperimentbecauseitistargeted,simple,challenging,scalableandfast.Byconstruction,anagentthatperformswellonthistaskhasmasteredsomekeypropertiesofdeepexploration.Oursummaryscoreprovidesa‘quickanddirty’waytocompareagentperformanceatahighlevel.Oursweepoverdi erentsizesNcanhelptopro-videempiricalevidenceofthescalingpropertiesofanalgorithmbeyondasimplepass/fail.Figure3presentsexampleoutputcomparingA2C,DQNandBootstrappedDQNonthis6 Q(s0,left)>Q(s0,right)<latexit sha1_base64="1LXzUxtf7xwlRf0H7N1UuC1ObIM=">AAAC73icjVHLSsNAFD2N73fVpZvBIqhISdzoSgpuXLZgW8FKSdJpHUyTOJmIpfgP7tyJW3/Arf6FuHNZ/8I70wg+EJ2Q5My555yZO+PFgUiUbb/krJHRsfGJyanpmdm5+YX84lItiVLp86ofBZE88tyEByLkVSVUwI9iyd2uF/C6d7av6/ULLhMRhYeqF/OTrtsJRVv4riKqmd+srCdNe4s1FL9U/YC31dUG22NfWCk6p0Q38wW7aJvBfgInA4USqwxeAZSj/DMaaCGCjxRdcIRQhAO4SOg5hgMbMXEn6BMnCQlT57jCNHlTUnFSuMSe0bdDs+OMDWmuMxPj9mmVgF5JToY18kSkk4T1aszUU5Os2d+y+yZT761Hfy/L6hKrcErsX74P5X99uheFNnZND4J6ig2ju/OzlNScit45+9SVooSYOI1bVJeEfeP8OGdmPInpXZ+ta+oDo9SsnvuZNsWb3iVdsPP9On+C2nbRsYtOxSmUShiOSaxgFet0nzso4QBlVCn7Gg94xJN1bt1Yt9bdUGrlMs8yvgzr/h2aoqGd</latexit> update(s0,left)<latexit sha1_base64="uZtnJrwEpKAjU+PUPB9B9AsuGuk=">AAAC5XicjVHLSsNAFD2Nr1pfUZdugkWoICVxo+4KblxWsA9pS0nSaQ1Ok5BMxFIKrty5E7f+gFv9FfEP9C+8M01BLaI3JDlz7j1n5s51Qu7FwjTfMtrM7Nz8QnYxt7S8srqmr29U4yCJXFZxAx5EdceOGfd8VhGe4KweRszuO5zVnMtjma9dsSj2Av9MDELW6ts93+t6ri2IautGU7BrMUzCji3YyCjEbXMv5TjritFuW8+bRVOFMQ2sFORLRzeQUQ70VzTRQQAXCfpg8CEIc9iI6WnAgomQuBaGxEWEPJVnGCFH2oSqGFXYxF7St0erRsr6tJaesVK7tAunNyKlgR3SBFQXEZa7GSqfKGfJ/uY9VJ7ybAP6O6lXn1iBC2L/0k0q/6uTvQh0cah68KinUDGyOzd1SdStyJMbX7oS5BASJ3GH8hFhVykn92woTax6l3drq/y7qpSsXLtpbYIPeUoasPVznNOgul+0zKJ1auVL5xhHFlvYRoHmeYASTlBGhbxv8YRnvGg97U671x7GpVom1WziW2iPnzF0nZg=</latexit> Q(s0,left)+bonus>Q(s0,right)<latexit sha1_base64="SAU/pL5gRNt9oxh5A4JsNOV0FJY=">AAAC/nicjVHLSsNAFD2N73fVpZvBIihKSdyoGym4cdmCVUGlJHFaB9MkzEzEUgr+iTt34tYfcKtLcedS/8I7YwQfiE5Icubcc87MnQnSSCjtuk8Fp69/YHBoeGR0bHxicqo4PbOrkkyGvB4mUSL3A1/xSMS8roWO+H4qud8OIr4XnG6Z+t4Zl0ok8Y7upPyo7bdi0RShr4lqFDdqi6rhrrBDzc91N+JN3Vtiy/k0SOJM9dgm+yKSonVCqkax5JZdO9hP4OWgVGG1l2cA1aT4iEMcI0GIDG1wxNCEI/hQ9BzAg4uUuCN0iZOEhK1z9DBK3oxUnBQ+saf0bdHsIGdjmptMZd0hrRLRK8nJsECehHSSsFmN2Xpmkw37W3bXZpq9degf5FltYjVOiP3L96H8r8/0otHEuu1BUE+pZUx3YZ6S2VMxO2efutKUkBJn8DHVJeHQOj/OmVmPsr2bs/Vt/cUqDWvmYa7N8Gp2SRfsfb/On2B3tey5Za/mlSoVvI9hzGEei3Sfa6hgG1XUKfsSd7jHg3PhXDnXzs271Cnknll8Gc7tG1fup94=</latexit>Published as a conference paper at ICLR 2022

experience buffer to update, which may not properly propagate the uncertainty. As a result, the
Q-values are not always optimistic, and induced action sequences may not be diverse. This issue is
pointed out in (Bai et al., 2021). Again, ϵ-greedy may be helpful to address this issue.

E.4 DISCUSSION OF NOISYNET

In this part, we explain the differences between HyperDQN and NoisyNet in detail. The following
discussion aims to provide insights about what HyperDQN can achieve while NoisyNet cannot. Note
that we are by no means criticizing NoisyNet. Instead, NoisyNet is simple and has strong empirical
performance. We hope the above discussion could provide intuitions (or explanations) why NoisyNet
succeeds or fails under different cases.

• First, as remarked in (Fortunato et al., 2018), NoisyNet is not ensured to approximate
the posterior distribution of parameters. Therefore, NoisyNet is not a typical Thompson
sampling based algorithm. The implication is that we may not use the well-known theory
(Osband et al., 2013; 2019; Zanette et al., 2020) to analyze NoisyNet.

• Second, NoisyNet re-samples a new policy every time step (Line 5 of Algorithm 1 in
(Fortunato et al., 2018)). Consequently, it does not implement deep exploration like RLSVI
(see (Osband, 2016b, Section 4.1)). This may explain the empirical result that NoisyNet can
not solve the deep sea task when the problem size is large than 20 (see Figure (Osband et al.,
2018, Figure 9)). Instead, BootDQN and HyperDQN can solve the deep sea task even if the
problem size is large than 20.

• Third, NoisyNet does not have the mechanism of “prior” (Osband et al., 2018) even though
it is randomized. As a consequence, NoisyNet cannot leverage an informative prior to
accelerate exploration when such information is available. In contrast, HyperDQN can
achieve this goal as discussed in Appendix E.6.

E.5 EXTENSION TO CONTINUOUS CONTROL

In this part, we briefly discuss how to extend the idea in this paper for continuous control tasks. We
also present some preliminary results to support this direction.

Following the idea presented in Section 4.3, we should leverage the hypermodel to capture the
posterior distribution of the Q-value function. Considering the standard actor-critic methods, we can
replace the vanilla critic with the one that is built by the hypermodel. In particular, we replace the
last layer of the critic with a hypermodel. In this way, each z corresponds to a specific critic from
the posterior distribution. To perform policy optimization, we notice that each actor (i.e., a policy
network) should be greedy to a specific critic with the same index z. To this end, the last layer of the
actor network is also built by a hypermodel. The architecture design is illustrated in Figure 23.

Figure 23: Illustration for HAC. In addition to a hypermodel in the critic network, there is also a
hypermodel for the actor network.

hidden, νcritic) be the parameter for the critic network and (θactor

Let (θcritic
hidden, νactor) be the parameter for the
actor network. Following the optimization framework in (Lillicrap et al., 2016; Haarnoja et al., 2018),
the training objective for the actor network is:

max
θactor
hidden,νactor

(cid:88)

(cid:88)

z∈ (cid:101)Z

s∈ (cid:101)D

Qθcritic

hidden,νcritic (s, az, z), with az ∼ πθactor

hidden,νactor (s; z),

(E.1)

34

(cid:68)(cid:70)(cid:87)(cid:82)(cid:85)(cid:3)(cid:81)(cid:72)(cid:87)(cid:90)(cid:82)(cid:85)(cid:78)(cid:70)(cid:85)(cid:76)(cid:87)(cid:76)(cid:70)(cid:3)(cid:81)(cid:72)(cid:87)(cid:90)(cid:82)(cid:85)(cid:78)Published as a conference paper at ICLR 2022

Algorithm 3 HyperActorCritic(HAC)

for episode k = 0, 1, 2, · · · do

generate a random vector z ∼ N (0, I).
instantiate an actor πθ(·; z).
for stage t = 0, 1, 2, · · · , T − 1 do

observe state st.
sample the action at ∼ πθ(st; z).
receive the next state st+1 and reward r(st, at).
sample ξ uniformly from unit hypersphere.
store (st, at, r, ξ, st+1) into the replay buffer D.
agent step n ← n + 1.
if mod (agent step n, train frequency M ) == 0 then

▷ Sampling

▷ Interaction

▷ Update

sample a mini-batch (cid:101)D of (s, a, r, ξ, s′) from the replay buffer D.
sample N random vectors z′
i: (cid:101)Z = {z′
optimize the critic network using the loss function (E.2) with (cid:101)D and (cid:101)Z.
optimize the actor network using the loss function (E.1) with (cid:101)D and (cid:101)Z.
update the critic target network with exponential moving average.

i=1.

i}N

end if

end for

end for

hidden,νcritic is the critic network and πθactor

where Qθcritic
,νactor is the actor network. In particular, objective
(E.1) states that we should sample actions from the actor network to maximize the Q-value function.
The difference with the traditional actor-critic methods is that we sample many actor networks
(indexed by z) to optimize.

θhidden

Similarly, the training objective for the critic network is:

min
θcritic
hidden,νcritic

(cid:88)

(cid:88)

z∈ (cid:101)Z

(s,a,r,s′,ξ)∈ (cid:101)D

(cid:16)

Qθcritic

hidden,νcritic(s, a, z) − σwξ⊤z −

(cid:16)

r + γQθcritic

hidden,νcritic (s′, a′

z, z)

(cid:17)(cid:17)2

,

(E.2)
with a′
hidden,νcritic is the target network. The goal of (E.2) is to
perform the temporal difference learning to optimize the critic. Note that the prior regularization term
is not appeared in (E.2) for easy presentation.

hidden,νactor (s′, z). In (E.2), Qθcritic

z ∼ πθactor

We call the above method as HyperActorCritic(HAC). Its implementation is outlined in Algorithm 3,
which shares many features with Algorithm 2.

Now, we consider the Cart Pole task from dm control13 (Tunyasuvunakool et al., 2020) as a
testbed; see Figure 24a . Detailed environment information is provided as follows: the dimension
of the state space is 5, the dimension of the action space is 1, the planning horizon is 1000 and the
reward is between 0 and 1. This environment is a standard platform to test the exploration efficiency
for continuous control algorithms because the agent can only obtain +1 reward when it succeeds and
obtain 0 reward otherwise.

We compare our extension (HAC) with the strong baseline SAC (Haarnoja et al., 2018). Even
though the policy entropy reward is used to guide exploration, SAC does not have the epistemic
uncertainty qualification like HAC. As a result, we expect SAC does not perform well for Cart Pole.
The numerical result is displayed in Figure 24b. We see that HAC outperforms SAC on this hard
exploration task.

E.6 WHEN AN INFORMATIVE PRIOR IS AVAILABLE

In this part, we briefly discuss the role of prior value functions in our formulation. In particular,
we argue that if an informative prior value function is available, the exploration efficiency can be
significantly improved.

13https://github.com/deepmind/dm_control

35

Published as a conference paper at ICLR 2022

(a) Illustration for Cart Pole.

(b) Episode return on Cart Pole.

Figure 24: Comparison of HAC and SAC on the hard exploration Cart Pole (Tunyasuvunakool et al.,
2020).

Here we show the learning curve with an informative prior in Figure 25. In particular, this informative
prior value function is obtained from the pre-trained model after 20M frames. We see that this
informative prior value function improves efficiency a lot. Note that the main goal of this experiment
is to illustrate that an informative prior could accelerate exploration. We will consider how to
automatically acquire an informative prior with human demonstrations (Hester et al., 2018; Sun et al.,
2018) in the future.

Figure 25: Comparison of HyperDQN with and without an informative prior on SuperMarioBros-1-3.

36

fromdm_control.suite.wrappersimportpixelsenv=suite.load("cartpole","swingup")#Replaceexistingfeaturesbypixelobservations:env_only_pixels=pixels.Wrapper(env)#Pixelobservationsinadditiontoexistingfeatures.env_plus_pixels=pixels.Wrapper(env,pixels_only=False)Rewardvisualisation:ModelsintheControlSuiteuseacommonsetofcoloursandtexturesforvisualuniformity.Asillustratedinthevideo,thisalsoallowsustomodifycoloursinproportiontothereward,providingaconvenientvisualcue.env=suite.load("fish","swim",task_kwargs,visualize_reward=True)6.2DomainsandTasksAdomainreferstoaphysicalmodel,whileataskreferstoaninstanceofthatmodelwithaparticularMDPstructure.Forexamplethediﬀerencebetweentheswingupandbalancetasksofthecartpoledomainiswhetherthepoleisini-tialisedpointingdownwardsorupwards,respectively.Insomecases,e.g.whenthemodelisprocedurallygenerated,diﬀerenttasksmighthavediﬀerentphysicalproper-ties.TasksintheControlSuitearecollatedintotuplesaccordingtopredeﬁnedtags.TasksusedforbenchmarkingareintheBENCHMARKINGtuple(Figure1),whilethosenotusedforbenchmarking(becausetheyareparticularlydiﬃcult,orbecausetheydonotconformtothestandardstructure)areintheEXTRAtuple.AllsuitetasksareaccessibleviatheALL_TASKStuple.Inthedomaindescriptionsbelow,namesarefollowedbythreeintegersspecifyingthedimensionsofthestate,controlandobservationspaces:Name(cid:16)dim(S),dim(A),dim(O)(cid:17).Pendulum(2,1,3):Theclassicinvertedpendulum.Thetorque-limitedactuatoris1/6thasstrongasrequiredtoliftthemassfrommo-tionlesshorizontal,necessitatingseveralswingstoswingupandbalance.Theswinguptaskhasasimplesparsereward:1whenthepoleiswithin30◦oftheverticalpositionand0otherwise.Acrobot(4,1,6):Theunderactuateddoublependulum,torqueap-pliedtothesecondjoint.Thegoalistoswingupandbalance.Despitebeinglow-dimensional,thisisnotaneasycontrolproblem.Thephysicalmodelconformsto(Coulom,2002)ratherthantheearlierSpong1995.Theswingupandswingup_sparsetaskshavesmoothandsparsere-wards,respectively.Cart-pole(4,1,5):Swingupandbalanceanunactuatedpolebyap-plyingforcestoacartatitsbase.ThephysicalmodelconformstoBartoetal.1983.Fourbenchmarkingtasks:inswingupandswingup_sparsethepolestartspointingdownwhileinbalanceandbalance_sparsethepolestartsneartheuprightposition.Cart-k-pole(2k+2,1,3k+2):Thecart-poledo-mainallowstoprocedurallyaddingmorepoles,connectedserially.Twonon-benchmarkingtasks,two_polesandthree_polesareavailable.230.000.250.500.751.00Number of Interactions (millions)0200400600800Episode ScoreCartPole-Swingup(Sparse Reward)HACSAC05101520Frame (millions)0150030004500600075009000Episode ScoreSuperMarioBros-1-3HyperDQNHyperDQN(with an informative prior)