# Introduction to neural posterior estimation

In contrast to classical parameter estimation codes like [Bilby](https://lscsoft.docs.ligo.org/bilby/index.html) and [LALInference](https://lscsoft.docs.ligo.org/lalsuite/lalinference/index.html), Dingo uses simulation-based (or likelihood-free) inference. The basic idea is to train a neural network to represent the Bayesian posterior over source parameters given the observed data. Training is based on simulated data rather than likelihood evaluations. [Neural posterior estimation (NPE)](https://arxiv.org/abs/1605.06376) combines the ideas of simulation-based inference with conditional neural density estimators.

## Normalizing flows

Normalizing flows provide a means to represent complicated probability distributions using neural networks, in a way that enables rapid sampling and density estimation. They represent the distribution in terms of a mapping (or flow) $f: u \to \theta$ on the sample space from a much simpler "base" distribution, which we take to be standard normal (of the same dimension as the parameter space). If $f$ is allowed to depend on observed data $d$ (denoted $f_d$) then the flow describes a conditional probability distribution $q(\theta | d)$. The PDF is given by the change of variables rule,

$$
q(\theta | d) = \mathcal{N}(0, 1)^D(f_d^{-1}(\theta)) \left| \det f_d^{-1} \right|,
$$ (eq:flow)

where $D$ is the dimensionality of the parameter space.

A normalizing flow must satisfy the following properties:
1. **Invertibility,** so that one can evaluate $f_d^{-1}(\theta)$ for any $\theta$. 
2. **Simple Jacobian determinant,** so that one can quickly evaluate $\det f_d^{-1}(\theta)$.

With these properties, one can quickly evaluate the right-hand side of {eq}`eq:flow` to obtain the density. Various types of normalizing flow have been constructed to satisfy these properties, typically as a composition of relatively simple transforms $f^{(j)}$. These relatively simple transforms are then parametrized by the output of a neural network. To sample $\theta \sim q(\theta|d)$, one samples $u \sim \mathcal N(0,1)^D$ and applies the flow in the forward direction.

For each flow step, Dingo uses a conditional coupling transform, meaning that half of the components are held fixed, and the other half transform elementwise, conditional on the untransformed components and the data,

$$
\begin{equation}
  f^{(j)}_{d,i}(u) =
  \begin{cases}
    u_i & \text{if } i \le D/2,\\
    f^{(j)}_i(u_i; u_{1:D/2},d) & \text{if } i > D/2.
  \end{cases}
\end{equation}
$$
If the elementwise functions $f^{(j)}_i$ are differentiable, then it follows automatically that we have a normalizing flow. We use a [neural spline flow](https://arxiv.org/abs/1906.04032), meaning that the functions $f^{(j)}_i$ are splines, which in turn are parametrized by neural network outputs (taking as input $(u_{1:D/2},d)$). Between each of these transforms, the parameters are randomly permuted, ensuring that the full flow is sufficiently flexible. Dingo uses the implementation of this entire structure provided by [nflows](https://github.com/bayesiains/nflows).


## Training

The conditional neural density estimator $q(\theta | d)$ is initialized randomly and must be trained to become a good approximation to the posterior $p(\theta | d)$. To achieve this, one must specify a target loss function to minimize. A reasonable starting point is to minimize the [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) of $p$ from $q$,

$$
D_{\text{KL}}(p \| q) = \int d\theta\, p(\theta | d) \log \frac{p(\theta | d)}{q(\theta | d)}.
$$

This measures a deviation between the two distributions, and is notably not symmetric. (We take the so-called "forward" KL divergence, which is "mass-covering".) Taking the expectation over data samples $d \sim p(d)$, and dropping the numerator from the $\log$ term (since it is independent of the network parameters), we arrive at the loss function

$$
\begin{align}
    L &= \int dd\, p(d) \int d\theta\, p(\theta | d) \left[ - \log q(\theta | d) \right]\\
    &=  \int d\theta\, p(\theta) \int dd\, p(d|\theta)\left[ - \log q(\theta | d) \right].
\end{align}
$$ (eq:loss)

On the second line we used Bayes' theorem $p(d) p(\theta | d) = p(\theta) p(d | \theta)$ to re-order the integrations. The loss may finally be approximated on a mini-batch of samples,

$$
L \approx - \frac{1}{N} \sum_{i=1}^N \log q(\theta^{(i)} | d^{(i)}),
$$

where the samples are drawn ancestrally in a two-step process:
1. **Sample from the prior,** $\theta^{(i)} \sim p(\theta)$,
2. **Simulate data,** $d^{(i)} \sim p(d | \theta^{(i)})$,

We then take the gradient of $L$ with respect to network parameters and minimize using the [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer.

Importantly, the process to generate training samples incorporates the **same information** as a standard (likelihood-based) sampler would use. Namely, the prior is incorporated by sampling parameters from it, and the likelihood is incorporated by simulating data. Bayes' theorem is incorporated in going from line 1 to line 2 in {eq}`eq:loss`. For gravitational waves, the likelihood is taken to be the probability that the residual when subtracting a signal $h(\theta)$ from $d$ is stationary Gaussian noise (with the measured PSD $S_{\text{n}}(f)$ in the detector). Likewise, to simulate data we generate a waveform $h(\theta^{(i)})$ and add a random noise realization $n \sim \mathcal N(0, S_\text{n}(f))$. Ultimately, however, the SBI approach is more flexible, since in principle one could add non-stationary or non-Gaussian noise, and train the network to reproduce the posterior, despite not having a tractable likelihood. See the section on training data for additional details of training for gravitational wave inference.

Intuitively, one way to understand NPE is simply that we are doing supervised deep learning---inferring parameter labels from examples---but allowing for the flexibility to produce a probabilistic answer. With this flexibility, the network learns to produce the Bayesian posterior.
