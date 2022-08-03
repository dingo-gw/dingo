# GNPE

GNPE (Gibbs- or Group-Equivariant Neural Posterior Estimation) is an algorithm that can generate significantly improved results by incorporating known physical symmetries into NPE.{footcite:p}`Dax:2021myb` The aim is to simplify the data seen by the network by using the symmetries to transform certain parameters to "standardized" values. This simplifies the learning task of the network. At inference time, the standardizing transform is initially unknown, so we use Gibbs sampling to simultaneously learn the transform (along with the rest of the parameters)  *and* apply it to simplify the data.

For gravitational waves, we use GNPE to standardize the times of arrival of the signal in the individual interferometers. (This corresponds to translations of the time of arrival at geocenter, and approximate sky rotations.) In frequency domain, time translations correspond to multiplication of the data by $e^{-2\pi i f \Delta t}$, and a standard NPE network would have to learn to interpret such transformations consistent with the prior from the data. We found this to be a challenging learning task, which limited inference performance on the other parameters. Instead, GNPE leverages our knowledge of the time translations to build a network that is only required to interpret a much narrower window of arrival times.

We now provide a brief description of the GNPE method. Readers more interested in getting started with GNPE may skip to [](#usage) below.

## Description of method

GNPE allows us to incorporate knowledge of **joint symmetries of data and parameters**. That is, if a parameter (e.g., coalescence time) is transformed by a certain amount ($\Delta t$), then there is a corresponding transformation of the data (multiplication by $e^{-2\pi i f \Delta t}$) such that the transformed data is equally likely to occur under the transformed parameter,

$$
p(t_c | d) = p(t_c + \Delta t | d\cdot e^{-2\pi i f \Delta t}).
$$

It is based on two ideas: 

### Gibbs + NPE

[Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) is an algorithm for obtaining samples from a joint distribution $p(x, y)$ if we are able to sample directly from each of the conditionals, $p(x|y)$ and $p(y|x)$. Starting from some point $y_0$, we construct a Markov chain $\{(x_i, y_i)\}$ by sampling

1. $x_i \sim p(x_i | y_{i-1})$,
2. $y_i \sim p(y_i | x_i)$,

and repeating until the chain is converged. The stationary distribution of the Markov chain is then $p(x, y)$.

```{figure} gibbs_figure.jpg
---
height: 300px
---    
Illustration of Gibbs sampling for a distribution $p(x, y)$.
```

Gibbs sampling can be combined with NPE by first introducing blurred "proxy" versions of a subset of parameters, which we denote $\hat\theta$ i.e., $\hat\theta \sim p(\hat\theta | \theta)$ where $p(\hat\theta | \theta)$ is defined by a blurring kernel. For example, for GWs we take $\hat t_I = t_I + \epsilon_I$, where $\epsilon_I \sim \text{Unif}(-1~\text{ms}, 1~\text{ms})$. We then train a network to model the posterior, but now conditioned also on $\hat \theta$, i.e., $p(\theta | d, \hat\theta)$. We can then apply Gibbs sampling to obtain samples from the joint distribution $p(\theta, \hat \theta | d)$, since we are able to sample individually from the conditional distributions:

* We can sample from $p(\hat\theta | \theta)$ since we defined the blurring kernel.
* We can sample from $p(\theta | d, \hat\theta)$ since we are modeling it using NPE.

Finally, we can drop $\hat \theta$ from the samples to obtain the desired posterior samples.

The trick now is that since $p(\theta | d, \hat\theta)$ is conditional on $\hat \theta$, we can apply any $\hat\theta$-dependent transformation to $d$. Returning to the time translations, $\hat t_I$ is a good approximation to $t_I$, so we apply the inverse time shift $d_I \to d_I\cdot e^{2 \pi i f \hat t_I}$, which brings $d_I$ into a close approximation to having coalescence time $0$ in each detector. This means that the network never sees any data with merger time further than $1~\text{ms}$ from $0$, greatly simplifying the learning task.

In practice, we generate many Monte Carlo chains in parallel---one for each desired sample and with different starting points---and keep only the final sample from each chain---rather than generating one long chain. Each individual chain in this ensemble is unlikely to converge, but if the individual chains are initialized from a distribution sufficiently close to $p(\hat \theta | d)$ then the collection of final samples from each chain should be a good approximation to samples from $p(\theta, \hat\theta|d)$.

### Group-equivariant NPE

So far we have described how Gibbs sampling together with NPE can simplify data by allowing any $\hat\theta$-dependent transformation of $d$, simplifying the data distribution. If we know the data and parameters to be equivariant under a particular transformation, however, we can go a step further and enforce this exactly. To do so, we simply drop the dependence of the neural density estimator on $\hat\theta$.

For gravitational waves, the overall time translation symmetry (in each detector) of the time of coalescence at geocenter is an exact symmetry, so we fully enforce this. The sky rotation, however, corresponds to an approximate symmetry: it shifts the time of coalescence in each detector, but a subleading effect is to change angle of incidence on a detector and hence the combination of polarizations observed. For this latter symmetry, we simply do not drop the proxy dependence.

```{tip}
GNPE is a generic method to incorporate symmetries into NPE:

* **Any** symmetry (exact or approximate) connecting data and parameters
* **Any** architecture, as it just requires (at most) conditioning on the proxy variables
```

As far as we are aware, GNPE is the only way to incorporate symmetries connecting data and parameters into architectures such as normalizing flows.



## Usage


### Training

To use GNPE for GW inference one must train **two** Dingo models:


1. An **initialization network** modeling $p(t_I | d)$. This gives the initial guess of the proxy variables for the staring point of the Gibbs sampler. Since this is only modeling two or three parameters and it does not need to give perfect results, this network can also be much smaller than typical Dingo networks.

   For an HL detector network, to infer *just* the detector coalescence times, set this in the train configuration. 
    ```yaml
    data:
      inference_parameters: [H1_time, L1_time]
    ```
2. A **main "GNPE" network**, conditional on the proxy variables, $p(\theta | d, \hat t_I)$. Implicitly in this expression, the data are transformed by the proxies, and the exact time-translation symmetry is enforced.

   To condition this network on the correct proxies, we configure it to use GNPE in the settings file.
    ```yaml
    data:
      gnpe_time_shifts:
        kernel: bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)
        exact_equiv: True
    ```  
   This sets the blurring kernel to be $\text{Unif}(-1~\text{ms}, 1~\text{ms})$ for all $\hat t_I$, and it specifies to enforce the overall time of coalescence symmetry exactly. Dingo will determine automatically from the `detectors` setting which proxy variables to condition on. 

Complete example config files for both networks are provided in the /examples folder.


### Inference

The inference script must be pointed to both trained networks in order to sample using GNPE.
```bash
dingo_analyze_event
  --model model
  --model_init model_init
  --gps_time_event gps_time_event
  --num_samples num_samples
  --num_gnpe_iterations num_gnpe_iterations
  --batch_size batch_size
```
The number of Gibbs iterations is also specified here (typically 30 is appropriate). This script will save the final samples from each Gibbs chain.



## The `GNPESampler` class

The inference script above uses the `GWSamplerGNPE` class, which is based on `GNPESampler`,
```{eval-rst}
.. autoclass:: dingo.core.samplers.GNPESampler
   :members:
   :inherited-members:
   :show-inheritance:
```
In addition to storing a `PosteriorModel`, a `GNPESampler` also stores a second `Sampler` instance, which is based on the initialization network.  When `run_sampler()` is called, it first generates samples from the initialization network, perturbs them with the kernel to obtain proxy samples, and then performs `num_iterations` Gibbs steps to obtain the final samples.

```{eval-rst}
.. footbibliography::
```