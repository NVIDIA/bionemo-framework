# Table of Contents

* [moco](#moco)
* [bionemo.moco.distributions](#mocodistributions)
* [bionemo.moco.distributions.prior.distribution](#mocodistributionspriordistribution)
* [bionemo.moco.distributions.prior.discrete.uniform](#mocodistributionspriordiscreteuniform)
* [bionemo.moco.distributions.prior.discrete.custom](#mocodistributionspriordiscretecustom)
* [bionemo.moco.distributions.prior.discrete](#mocodistributionspriordiscrete)
* [bionemo.moco.distributions.prior.discrete.mask](#mocodistributionspriordiscretemask)
* [bionemo.moco.distributions.prior.continuous.harmonic](#mocodistributionspriorcontinuousharmonic)
* [bionemo.moco.distributions.prior.continuous](#mocodistributionspriorcontinuous)
* [bionemo.moco.distributions.prior.continuous.gaussian](#mocodistributionspriorcontinuousgaussian)
* [bionemo.moco.distributions.prior.continuous.utils](#mocodistributionspriorcontinuousutils)
* [bionemo.moco.distributions.prior](#mocodistributionsprior)
* [bionemo.moco.distributions.time.distribution](#mocodistributionstimedistribution)
* [bionemo.moco.distributions.time.uniform](#mocodistributionstimeuniform)
* [bionemo.moco.distributions.time.logit\_normal](#mocodistributionstimelogit_normal)
* [bionemo.moco.distributions.time](#mocodistributionstime)
* [bionemo.moco.distributions.time.beta](#mocodistributionstimebeta)
* [bionemo.moco.distributions.time.utils](#mocodistributionstimeutils)
* [bionemo.moco.schedules.discrete\_noise\_schedules](#mocoschedulesdiscrete_noise_schedules)
* [bionemo.moco.schedules.noise.continuous\_snr\_transforms](#mocoschedulesnoisecontinuous_snr_transforms)
* [bionemo.moco.schedules.noise.discrete\_noise\_schedules](#mocoschedulesnoisediscrete_noise_schedules)
* [bionemo.moco.schedules.noise](#mocoschedulesnoise)
* [bionemo.moco.schedules.noise.continuous\_noise\_transforms](#mocoschedulesnoisecontinuous_noise_transforms)
* [bionemo.moco.schedules](#mocoschedules)
* [bionemo.moco.schedules.utils](#mocoschedulesutils)
* [bionemo.moco.schedules.inference\_time\_schedules](#mocoschedulesinference_time_schedules)
* [bionemo.moco.interpolants.continuous\_time.discrete](#mocointerpolantscontinuous_timediscrete)
* [bionemo.moco.interpolants.continuous\_time.discrete.mdlm](#mocointerpolantscontinuous_timediscretemdlm)
* [bionemo.moco.interpolants.continuous\_time.discrete.discrete\_flow\_matching](#mocointerpolantscontinuous_timediscretediscrete_flow_matching)
* [bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport.ot\_types](#mocointerpolantscontinuous_timecontinuousoptimal_transportot_types)
* [bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport.ot\_sampler](#mocointerpolantscontinuous_timecontinuousoptimal_transportot_sampler)
* [bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport.equivariant\_ot\_sampler](#mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_sampler)
* [bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport.kabsch\_augmentation](#mocointerpolantscontinuous_timecontinuousoptimal_transportkabsch_augmentation)
* [bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport](#mocointerpolantscontinuous_timecontinuousoptimal_transport)
* [bionemo.moco.interpolants.continuous\_time.continuous](#mocointerpolantscontinuous_timecontinuous)
* [bionemo.moco.interpolants.continuous\_time.continuous.vdm](#mocointerpolantscontinuous_timecontinuousvdm)
* [bionemo.moco.interpolants.continuous\_time.continuous.continuous\_flow\_matching](#mocointerpolantscontinuous_timecontinuouscontinuous_flow_matching)
* [bionemo.moco.interpolants.continuous\_time](#mocointerpolantscontinuous_time)
* [bionemo.moco.interpolants](#mocointerpolants)
* [bionemo.moco.interpolants.batch\_augmentation](#mocointerpolantsbatch_augmentation)
* [bionemo.moco.interpolants.discrete\_time.discrete.d3pm](#mocointerpolantsdiscrete_timediscreted3pm)
* [bionemo.moco.interpolants.discrete\_time.discrete](#mocointerpolantsdiscrete_timediscrete)
* [bionemo.moco.interpolants.discrete\_time.continuous.ddpm](#mocointerpolantsdiscrete_timecontinuousddpm)
* [bionemo.moco.interpolants.discrete\_time.continuous](#mocointerpolantsdiscrete_timecontinuous)
* [bionemo.moco.interpolants.discrete\_time](#mocointerpolantsdiscrete_time)
* [bionemo.moco.interpolants.discrete\_time.utils](#mocointerpolantsdiscrete_timeutils)
* [bionemo.moco.interpolants.base\_interpolant](#mocointerpolantsbase_interpolant)

<a id="moco"></a>

# moco

<a id="mocodistributions"></a>

# bionemo.moco.distributions

<a id="mocodistributionspriordistribution"></a>

# bionemo.moco.distributions.prior.distribution

<a id="mocodistributionspriordistributionPriorDistribution"></a>

## PriorDistribution Objects

```python
class PriorDistribution(ABC)
```

An abstract base class representing a prior distribution.

<a id="mocodistributionspriordistributionPriorDistributionsample"></a>

#### sample

```python
@abstractmethod
def sample(shape: Tuple,
           mask: Optional[Tensor] = None,
           device: Union[str, torch.device] = "cpu") -> Tensor
```

Generates a specified number of samples from the time distribution.

**Arguments**:

- `shape` _Tuple_ - The shape of the samples to generate.
- `mask` _Optional[Tensor], optional_ - A tensor indicating which samples should be masked. Defaults to None.
- `device` _str, optional_ - The device on which to generate the samples. Defaults to "cpu".


**Returns**:

- `Float` - A tensor of samples.

<a id="mocodistributionspriordistributionDiscretePriorDistribution"></a>

## DiscretePriorDistribution Objects

```python
class DiscretePriorDistribution(PriorDistribution)
```

An abstract base class representing a discrete prior distribution.

<a id="mocodistributionspriordistributionDiscretePriorDistribution__init__"></a>

#### \_\_init\_\_

```python
def __init__(num_classes: int, prior_dist: Tensor)
```

Initializes a DiscretePriorDistribution instance.

**Arguments**:

- `num_classes` _int_ - The number of classes in the discrete distribution.
- `prior_dist` _Tensor_ - The prior distribution over the classes.


**Returns**:

  None

<a id="mocodistributionspriordistributionDiscretePriorDistributionget_num_classes"></a>

#### get\_num\_classes

```python
def get_num_classes() -> int
```

Getter for num_classes.

<a id="mocodistributionspriordistributionDiscretePriorDistributionget_prior_dist"></a>

#### get\_prior\_dist

```python
def get_prior_dist() -> Tensor
```

Getter for prior_dist.

<a id="mocodistributionspriordiscreteuniform"></a>

# bionemo.moco.distributions.prior.discrete.uniform

<a id="mocodistributionspriordiscreteuniformDiscreteUniformPrior"></a>

## DiscreteUniformPrior Objects

```python
class DiscreteUniformPrior(DiscretePriorDistribution)
```

A subclass representing a discrete uniform prior distribution.

<a id="mocodistributionspriordiscreteuniformDiscreteUniformPrior__init__"></a>

#### \_\_init\_\_

```python
def __init__(num_classes: int = 10) -> None
```

Initializes a discrete uniform prior distribution.

**Arguments**:

- `num_classes` _int_ - The number of classes in the discrete uniform distribution. Defaults to 10.

<a id="mocodistributionspriordiscreteuniformDiscreteUniformPriorsample"></a>

#### sample

```python
def sample(shape: Tuple,
           mask: Optional[Tensor] = None,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None) -> Tensor
```

Generates a specified number of samples.

**Arguments**:

- `shape` _Tuple_ - The shape of the samples to generate.
- `device` _str_ - cpu or gpu.
- `mask` _Optional[Tensor]_ - An optional mask to apply to the samples. Defaults to None.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

- `Float` - A tensor of samples.

<a id="mocodistributionspriordiscretecustom"></a>

# bionemo.moco.distributions.prior.discrete.custom

<a id="mocodistributionspriordiscretecustomDiscreteCustomPrior"></a>

## DiscreteCustomPrior Objects

```python
class DiscreteCustomPrior(DiscretePriorDistribution)
```

A subclass representing a discrete custom prior distribution.

This class allows for the creation of a prior distribution with a custom
probability mass function defined by the `prior_dist` tensor. For example if my data has 4 classes and I want [.3, .2, .4, .1] as the probabilities of the 4 classes.

<a id="mocodistributionspriordiscretecustomDiscreteCustomPrior__init__"></a>

#### \_\_init\_\_

```python
def __init__(prior_dist: Tensor, num_classes: int = 10) -> None
```

Initializes a DiscreteCustomPrior distribution.

**Arguments**:

- `prior_dist` - A tensor representing the probability mass function of the prior distribution.
- `num_classes` - The number of classes in the prior distribution. Defaults to 10.


**Notes**:

  The `prior_dist` tensor should have a sum close to 1.0, as it represents a probability mass function.

<a id="mocodistributionspriordiscretecustomDiscreteCustomPriorsample"></a>

#### sample

```python
def sample(shape: Tuple,
           mask: Optional[Tensor] = None,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None) -> Tensor
```

Samples from the discrete custom prior distribution.

**Arguments**:

- `shape` - A tuple specifying the shape of the samples to generate.
- `mask` - An optional tensor mask to apply to the samples, broadcastable to the sample shape. Defaults to None.
- `device` - The device on which to generate the samples, specified as a string or a :class:`torch.device`. Defaults to "cpu".
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

  A tensor of samples drawn from the prior distribution.

<a id="mocodistributionspriordiscrete"></a>

# bionemo.moco.distributions.prior.discrete

<a id="mocodistributionspriordiscretemask"></a>

# bionemo.moco.distributions.prior.discrete.mask

<a id="mocodistributionspriordiscretemaskDiscreteMaskedPrior"></a>

## DiscreteMaskedPrior Objects

```python
class DiscreteMaskedPrior(DiscretePriorDistribution)
```

A subclass representing a Discrete Masked prior distribution.

<a id="mocodistributionspriordiscretemaskDiscreteMaskedPrior__init__"></a>

#### \_\_init\_\_

```python
def __init__(num_classes: int = 10,
             mask_dim: Optional[int] = None,
             inclusive: bool = True) -> None
```

Discrete Masked prior distribution.

Theres 3 ways I can think of defining the problem that are hard to mesh together.

1. [..., M, ....] inclusive anywhere --> exisiting LLM tokenizer where the mask has a specific location not at the end
2. [......, M] inclusive on end --> mask_dim = None with inclusive set to True default stick on the end
3. [.....] + [M] exclusive --> the number of classes representes the number of data classes and one wishes to add a separate MASK dimension.
- Note the pad_sample function is provided to help add this extra external dimension.

**Arguments**:

- `num_classes` _int_ - The number of classes in the distribution. Defaults to 10.
- `mask_dim` _int_ - The index for the mask token. Defaults to num_classes - 1 if inclusive or num_classes if exclusive.
- `inclusive` _bool_ - Whether the mask is included in the specified number of classes.
  If True, the mask is considered as one of the classes.
  If False, the mask is considered as an additional class. Defaults to True.

<a id="mocodistributionspriordiscretemaskDiscreteMaskedPriorsample"></a>

#### sample

```python
def sample(shape: Tuple,
           mask: Optional[Tensor] = None,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None) -> Tensor
```

Generates a specified number of samples.

**Arguments**:

- `shape` _Tuple_ - The shape of the samples to generate.
- `device` _str_ - cpu or gpu.
- `mask` _Optional[Tensor]_ - An optional mask to apply to the samples. Defaults to None.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

- `Float` - A tensor of samples.

<a id="mocodistributionspriordiscretemaskDiscreteMaskedPrioris_masked"></a>

#### is\_masked

```python
def is_masked(sample: Tensor) -> Tensor
```

Creates a mask for whether a state is masked.

**Arguments**:

- `sample` _Tensor_ - The sample to check.


**Returns**:

- `Tensor` - A float tensor indicating whether the sample is masked.

<a id="mocodistributionspriordiscretemaskDiscreteMaskedPriorpad_sample"></a>

#### pad\_sample

```python
def pad_sample(sample: Tensor) -> Tensor
```

Pads the input sample with zeros along the last dimension.

**Arguments**:

- `sample` _Tensor_ - The input sample to be padded.


**Returns**:

- `Tensor` - The padded sample.

<a id="mocodistributionspriorcontinuousharmonic"></a>

# bionemo.moco.distributions.prior.continuous.harmonic

<a id="mocodistributionspriorcontinuousharmonicLinearHarmonicPrior"></a>

## LinearHarmonicPrior Objects

```python
class LinearHarmonicPrior(PriorDistribution)
```

A subclass representing a Linear Harmonic prior distribution from Jit et al. https://arxiv.org/abs/2304.02198.

<a id="mocodistributionspriorcontinuousharmonicLinearHarmonicPrior__init__"></a>

#### \_\_init\_\_

```python
def __init__(distance: Float = 3.8,
             length: Optional[int] = None,
             center: Bool = False,
             rng_generator: Optional[torch.Generator] = None,
             device: Union[str, torch.device] = "cpu") -> None
```

Linear Harmonic prior distribution.

**Arguments**:

- `distance` _Float_ - RMS distance between adjacent points in the line graph.
- `length` _Optional[int]_ - The number of points in a batch.
- `center` _bool_ - Whether to center the samples around the mean. Defaults to False.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocodistributionspriorcontinuousharmonicLinearHarmonicPriorsample"></a>

#### sample

```python
def sample(shape: Tuple,
           mask: Optional[Tensor] = None,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None) -> Tensor
```

Generates a specified number of samples from the Harmonic prior distribution.

**Arguments**:

- `shape` _Tuple_ - The shape of the samples to generate.
- `device` _str_ - cpu or gpu.
- `mask` _Optional[Tensor]_ - An optional mask to apply to the samples. Defaults to None.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

- `Float` - A tensor of samples.

<a id="mocodistributionspriorcontinuous"></a>

# bionemo.moco.distributions.prior.continuous

<a id="mocodistributionspriorcontinuousgaussian"></a>

# bionemo.moco.distributions.prior.continuous.gaussian

<a id="mocodistributionspriorcontinuousgaussianGaussianPrior"></a>

## GaussianPrior Objects

```python
class GaussianPrior(PriorDistribution)
```

A subclass representing a Gaussian prior distribution.

<a id="mocodistributionspriorcontinuousgaussianGaussianPrior__init__"></a>

#### \_\_init\_\_

```python
def __init__(mean: Float = 0.0,
             std: Float = 1.0,
             center: Bool = False,
             rng_generator: Optional[torch.Generator] = None) -> None
```

Gaussian prior distribution.

**Arguments**:

- `mean` _Float_ - The mean of the Gaussian distribution. Defaults to 0.0.
- `std` _Float_ - The standard deviation of the Gaussian distribution. Defaults to 1.0.
- `center` _bool_ - Whether to center the samples around the mean. Defaults to False.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocodistributionspriorcontinuousgaussianGaussianPriorsample"></a>

#### sample

```python
def sample(shape: Tuple,
           mask: Optional[Tensor] = None,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None) -> Tensor
```

Generates a specified number of samples from the Gaussian prior distribution.

**Arguments**:

- `shape` _Tuple_ - The shape of the samples to generate.
- `device` _str_ - cpu or gpu.
- `mask` _Optional[Tensor]_ - An optional mask to apply to the samples. Defaults to None.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

- `Float` - A tensor of samples.

<a id="mocodistributionspriorcontinuousutils"></a>

# bionemo.moco.distributions.prior.continuous.utils

<a id="mocodistributionspriorcontinuousutilsremove_center_of_mass"></a>

#### remove\_center\_of\_mass

```python
def remove_center_of_mass(data: Tensor,
                          mask: Optional[Tensor] = None) -> Tensor
```

Calculates the center of mass (CoM) of the given data.

**Arguments**:

- `data` - The input data with shape (..., nodes, features).
- `mask` - An optional binary mask to apply to the data with shape (..., nodes) to mask out interaction from CoM calculation. Defaults to None.


**Returns**:

  The CoM of the data with shape (..., 1, features).

<a id="mocodistributionsprior"></a>

# bionemo.moco.distributions.prior

<a id="mocodistributionstimedistribution"></a>

# bionemo.moco.distributions.time.distribution

<a id="mocodistributionstimedistributionTimeDistribution"></a>

## TimeDistribution Objects

```python
class TimeDistribution(ABC)
```

An abstract base class representing a time distribution.

**Arguments**:

- `discrete_time` _Bool_ - Whether the time is discrete.
- `nsteps` _Optional[int]_ - Number of nsteps for discretization.
- `min_t` _Optional[Float]_ - Min continuous time.
- `max_t` _Optional[Float]_ - Max continuous time.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocodistributionstimedistributionTimeDistribution__init__"></a>

#### \_\_init\_\_

```python
def __init__(discrete_time: Bool = False,
             nsteps: Optional[int] = None,
             min_t: Optional[Float] = None,
             max_t: Optional[Float] = None,
             rng_generator: Optional[torch.Generator] = None)
```

Initializes a TimeDistribution object.

<a id="mocodistributionstimedistributionTimeDistributionsample"></a>

#### sample

```python
@abstractmethod
def sample(n_samples: int,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None) -> Float
```

Generates a specified number of samples from the time distribution.

**Arguments**:

- `n_samples` _int_ - The number of samples to generate.
- `device` _str_ - cpu or gpu.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

- `Float` - A list or array of samples.

<a id="mocodistributionstimedistributionMixTimeDistribution"></a>

## MixTimeDistribution Objects

```python
class MixTimeDistribution()
```

An abstract base class representing a mixed time distribution.

uniform_dist = UniformTimeDistribution(min_t=0.0, max_t=1.0, discrete_time=False)
beta_dist = BetaTimeDistribution(min_t=0.0, max_t=1.0, discrete_time=False, p1=2.0, p2=1.0)
mix_dist = MixTimeDistribution(uniform_dist, beta_dist, mix_fraction=0.5)

<a id="mocodistributionstimedistributionMixTimeDistribution__init__"></a>

#### \_\_init\_\_

```python
def __init__(dist1: TimeDistribution, dist2: TimeDistribution,
             mix_fraction: Float)
```

Initializes a MixTimeDistribution object.

**Arguments**:

- `dist1` _TimeDistribution_ - The first time distribution.
- `dist2` _TimeDistribution_ - The second time distribution.
- `mix_fraction` _Float_ - The fraction of samples to draw from dist1. Must be between 0 and 1.

<a id="mocodistributionstimedistributionMixTimeDistributionsample"></a>

#### sample

```python
def sample(n_samples: int,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None) -> Float
```

Generates a specified number of samples from the mixed time distribution.

**Arguments**:

- `n_samples` _int_ - The number of samples to generate.
- `device` _str_ - cpu or gpu.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

- `Float` - A list or array of samples.

<a id="mocodistributionstimeuniform"></a>

# bionemo.moco.distributions.time.uniform

<a id="mocodistributionstimeuniformUniformTimeDistribution"></a>

## UniformTimeDistribution Objects

```python
class UniformTimeDistribution(TimeDistribution)
```

A class representing a uniform time distribution.

<a id="mocodistributionstimeuniformUniformTimeDistribution__init__"></a>

#### \_\_init\_\_

```python
def __init__(min_t: Float = 0.0,
             max_t: Float = 1.0,
             discrete_time: Bool = False,
             nsteps: Optional[int] = None,
             rng_generator: Optional[torch.Generator] = None)
```

Initializes a UniformTimeDistribution object.

**Arguments**:

- `min_t` _Float_ - The minimum time value.
- `max_t` _Float_ - The maximum time value.
- `discrete_time` _Bool_ - Whether the time is discrete.
- `nsteps` _Optional[int]_ - Number of nsteps for discretization.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocodistributionstimeuniformUniformTimeDistributionsample"></a>

#### sample

```python
def sample(n_samples: int,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None)
```

Generates a specified number of samples from the uniform time distribution.

**Arguments**:

- `n_samples` _int_ - The number of samples to generate.
- `device` _str_ - cpu or gpu.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

  A tensor of samples.

<a id="mocodistributionstimeuniformSymmetricUniformTimeDistribution"></a>

## SymmetricUniformTimeDistribution Objects

```python
class SymmetricUniformTimeDistribution(TimeDistribution)
```

A class representing a uniform time distribution.

<a id="mocodistributionstimeuniformSymmetricUniformTimeDistribution__init__"></a>

#### \_\_init\_\_

```python
def __init__(min_t: Float = 0.0,
             max_t: Float = 1.0,
             discrete_time: Bool = False,
             nsteps: Optional[int] = None,
             rng_generator: Optional[torch.Generator] = None)
```

Initializes a UniformTimeDistribution object.

**Arguments**:

- `min_t` _Float_ - The minimum time value.
- `max_t` _Float_ - The maximum time value.
- `discrete_time` _Bool_ - Whether the time is discrete.
- `nsteps` _Optional[int]_ - Number of nsteps for discretization.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocodistributionstimeuniformSymmetricUniformTimeDistributionsample"></a>

#### sample

```python
def sample(n_samples: int,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None)
```

Generates a specified number of samples from the uniform time distribution.

**Arguments**:

- `n_samples` _int_ - The number of samples to generate.
- `device` _str_ - cpu or gpu.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

  A tensor of samples.

<a id="mocodistributionstimelogit_normal"></a>

# bionemo.moco.distributions.time.logit\_normal

<a id="mocodistributionstimelogit_normalLogitNormalTimeDistribution"></a>

## LogitNormalTimeDistribution Objects

```python
class LogitNormalTimeDistribution(TimeDistribution)
```

A class representing a logit normal time distribution.

<a id="mocodistributionstimelogit_normalLogitNormalTimeDistribution__init__"></a>

#### \_\_init\_\_

```python
def __init__(p1: Float = 0.0,
             p2: Float = 1.0,
             min_t: Float = 0.0,
             max_t: Float = 1.0,
             discrete_time: Bool = False,
             nsteps: Optional[int] = None,
             rng_generator: Optional[torch.Generator] = None)
```

Initializes a BetaTimeDistribution object.

**Arguments**:

- `p1` _Float_ - The first shape parameter of the logit normal distribution i.e. the mean.
- `p2` _Float_ - The second shape parameter of the logit normal distribution i.e. the std.
- `min_t` _Float_ - The minimum time value.
- `max_t` _Float_ - The maximum time value.
- `discrete_time` _Bool_ - Whether the time is discrete.
- `nsteps` _Optional[int]_ - Number of nsteps for discretization.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocodistributionstimelogit_normalLogitNormalTimeDistributionsample"></a>

#### sample

```python
def sample(n_samples: int,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None)
```

Generates a specified number of samples from the uniform time distribution.

**Arguments**:

- `n_samples` _int_ - The number of samples to generate.
- `device` _str_ - cpu or gpu.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

  A tensor of samples.

<a id="mocodistributionstime"></a>

# bionemo.moco.distributions.time

<a id="mocodistributionstimebeta"></a>

# bionemo.moco.distributions.time.beta

<a id="mocodistributionstimebetaBetaTimeDistribution"></a>

## BetaTimeDistribution Objects

```python
class BetaTimeDistribution(TimeDistribution)
```

A class representing a beta time distribution.

<a id="mocodistributionstimebetaBetaTimeDistribution__init__"></a>

#### \_\_init\_\_

```python
def __init__(p1: Float = 2.0,
             p2: Float = 1.0,
             min_t: Float = 0.0,
             max_t: Float = 1.0,
             discrete_time: Bool = False,
             nsteps: Optional[int] = None,
             rng_generator: Optional[torch.Generator] = None)
```

Initializes a BetaTimeDistribution object.

**Arguments**:

- `p1` _Float_ - The first shape parameter of the beta distribution.
- `p2` _Float_ - The second shape parameter of the beta distribution.
- `min_t` _Float_ - The minimum time value.
- `max_t` _Float_ - The maximum time value.
- `discrete_time` _Bool_ - Whether the time is discrete.
- `nsteps` _Optional[int]_ - Number of nsteps for discretization.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocodistributionstimebetaBetaTimeDistributionsample"></a>

#### sample

```python
def sample(n_samples: int,
           device: Union[str, torch.device] = "cpu",
           rng_generator: Optional[torch.Generator] = None)
```

Generates a specified number of samples from the uniform time distribution.

**Arguments**:

- `n_samples` _int_ - The number of samples to generate.
- `device` _str_ - cpu or gpu.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.


**Returns**:

  A tensor of samples.

<a id="mocodistributionstimeutils"></a>

# bionemo.moco.distributions.time.utils

<a id="mocodistributionstimeutilsfloat_time_to_index"></a>

#### float\_time\_to\_index

```python
def float_time_to_index(time: torch.Tensor,
                        num_time_steps: int) -> torch.Tensor
```

Convert a float time value to a time index.

**Arguments**:

- `time` _torch.Tensor_ - A tensor of float time values in the range [0, 1].
- `num_time_steps` _int_ - The number of discrete time steps.


**Returns**:

- `torch.Tensor` - A tensor of time indices corresponding to the input float time values.

<a id="mocoschedulesdiscrete_noise_schedules"></a>

# bionemo.moco.schedules.discrete\_noise\_schedules

<a id="mocoschedulesdiscrete_noise_schedulesDiscreteNoiseSchedule"></a>

## DiscreteNoiseSchedule Objects

```python
class DiscreteNoiseSchedule(ABC)
```

A base class for discrete schedules. No matter the definition this class returns objects using a unified direction of time.

<a id="mocoschedulesdiscrete_noise_schedulesDiscreteNoiseSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int, direction: TimeDirection)
```

Initialize the DiscreteNoiseSchedule.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None, uses the value from initialization.
- `direction` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).

<a id="mocoschedulesdiscrete_noise_schedulesDiscreteNoiseSchedulegenerate_schedule"></a>

#### generate\_schedule

```python
def generate_schedule(nsteps: Optional[int] = None,
                      device: Union[str, torch.device] = "cpu",
                      synchronize: Optional[TimeDirection] = None) -> Tensor
```

Public wrapper to generate the time schedule as a tensor.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None, uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").
- `synchronize` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).


**Returns**:

- `Tensor` - A tensor of time steps + 1 unless full is False.

<a id="mocoschedulesdiscrete_noise_schedulesDiscreteNoiseSchedulecalculate_derivative"></a>

#### calculate\_derivative

```python
def calculate_derivative(
        nsteps: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None) -> Tensor
```

Calculate the time derivative of the schedule.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None, uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").
- `synchronize` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).


**Returns**:

- `Tensor` - A tensor representing the time derivative of the schedule.


**Raises**:

- `NotImplementedError` - If the derivative calculation is not implemented for this schedule.

<a id="mocoschedulesdiscrete_noise_schedulesDiscreteCosineNoiseSchedule"></a>

## DiscreteCosineNoiseSchedule Objects

```python
class DiscreteCosineNoiseSchedule(DiscreteNoiseSchedule)
```

A cosine noise schedule for Diffusion Models.

<a id="mocoschedulesdiscrete_noise_schedulesDiscreteCosineNoiseSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int, nu: Float = 1.0, s: Float = 0.008)
```

Initialize the CosineNoiseSchedule.

**Arguments**:

- `nsteps` _int_ - Number of time steps.
- `nu` _Optional[Float]_ - Hyperparameter for the cosine schedule (default is 1.0).
- `s` _Optional[Float]_ - Hyperparameter for the cosine schedule (default is 0.008).

<a id="mocoschedulesnoisecontinuous_snr_transforms"></a>

# bionemo.moco.schedules.noise.continuous\_snr\_transforms

<a id="mocoschedulesnoisecontinuous_snr_transformslog"></a>

#### log

```python
def log(t, eps=1e-20)
```

Compute the natural logarithm of a tensor, clamping values to avoid numerical instability.

**Arguments**:

- `t` _Tensor_ - The input tensor.
- `eps` _float, optional_ - The minimum value to clamp the input tensor (default is 1e-20).


**Returns**:

- `Tensor` - The natural logarithm of the input tensor.

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransform"></a>

## ContinuousSNRTransform Objects

```python
class ContinuousSNRTransform(ABC)
```

A base class for continuous SNR schedules.

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransform__init__"></a>

#### \_\_init\_\_

```python
def __init__(direction: TimeDirection)
```

Initialize the DiscreteNoiseSchedule.

**Arguments**:

- `direction` _TimeDirection_ - required this defines in which direction the scheduler was built

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransformcalculate_log_snr"></a>

#### calculate\_log\_snr

```python
def calculate_log_snr(t: Tensor,
                      device: Union[str, torch.device] = "cpu",
                      synchronize: Optional[TimeDirection] = None) -> Tensor
```

Public wrapper to generate the time schedule as a tensor.

**Arguments**:

- `t` _Tensor_ - The input tensor representing the time steps, with values ranging from 0 to 1.
- `device` _Optional[str]_ - The device to place the schedule on. Defaults to "cpu".
- `synchronize` _optional[TimeDirection]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
  this parameter allows to flip the direction to match the specified one. Defaults to None.


**Returns**:

- `Tensor` - A tensor representing the log signal-to-noise (SNR) ratio for the given time steps.

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransformlog_snr_to_alphas_sigmas"></a>

#### log\_snr\_to\_alphas\_sigmas

```python
def log_snr_to_alphas_sigmas(log_snr: Tensor) -> Tuple[Tensor, Tensor]
```

Converts log signal-to-noise ratio (SNR) to alpha and sigma values.

**Arguments**:

- `log_snr` _Tensor_ - The input log SNR tensor.


**Returns**:

  tuple[Tensor, Tensor]: A tuple containing the squared root of alpha and sigma values.

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransformderivative"></a>

#### derivative

```python
def derivative(t: Tensor, func: Callable) -> Tensor
```

Compute derivative of a function, it supports bached single variable inputs.

**Arguments**:

- `t` _Tensor_ - time variable at which derivatives are taken
- `func` _Callable_ - function for derivative calculation


**Returns**:

- `Tensor` - derivative that is detached from the computational graph

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransformcalculate_general_sde_terms"></a>

#### calculate\_general\_sde\_terms

```python
def calculate_general_sde_terms(t)
```

Compute the general SDE terms for a given time step t.

**Arguments**:

- `t` _Tensor_ - The input tensor representing the time step.


**Returns**:

  tuple[Tensor, Tensor]: A tuple containing the drift term f_t and the diffusion term g_t_2.


**Notes**:

  This method computes the drift and diffusion terms of the general SDE, which can be used to simulate the stochastic process.
  The drift term represents the deterministic part of the process, while the diffusion term represents the stochastic part.

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransformcalculate_beta"></a>

#### calculate\_beta

```python
def calculate_beta(t)
```

Compute the drift coefficient for the OU process of the form $dx = -\frac{1}{2} \beta(t) x dt + sqrt(beta(t)) dw_t$.

beta = d/dt log(alpha**2) = 2 * 1/alpha * d/dt(alpha)

**Arguments**:

- `t` _Union[float, Tensor]_ - t in [0, 1]


**Returns**:

- `Tensor` - beta(t)

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransformcalculate_alpha_log_snr"></a>

#### calculate\_alpha\_log\_snr

```python
def calculate_alpha_log_snr(log_snr: Tensor) -> Tensor
```

Compute alpha values based on the log SNR.

**Arguments**:

- `log_snr` _Tensor_ - The input tensor representing the log signal-to-noise ratio.


**Returns**:

- `Tensor` - A tensor representing the alpha values for the given log SNR.


**Notes**:

  This method computes alpha values as the square root of the sigmoid of the log SNR.

<a id="mocoschedulesnoisecontinuous_snr_transformsContinuousSNRTransformcalculate_alpha_t"></a>

#### calculate\_alpha\_t

```python
def calculate_alpha_t(t: Tensor) -> Tensor
```

Compute alpha values based on the log SNR schedule.

**Arguments**:

- `t` _Tensor_ - The input tensor representing the time steps.


**Returns**:

- `Tensor` - A tensor representing the alpha values for the given time steps.


**Notes**:

  This method computes alpha values as the square root of the sigmoid of the log SNR.

<a id="mocoschedulesnoisecontinuous_snr_transformsCosineSNRTransform"></a>

## CosineSNRTransform Objects

```python
class CosineSNRTransform(ContinuousSNRTransform)
```

A cosine SNR schedule.

**Arguments**:

- `nu` _Optional[Float]_ - Hyperparameter for the cosine schedule exponent (default is 1.0).
- `s` _Optional[Float]_ - Hyperparameter for the cosine schedule shift (default is 0.008).

<a id="mocoschedulesnoisecontinuous_snr_transformsCosineSNRTransform__init__"></a>

#### \_\_init\_\_

```python
def __init__(nu: Float = 1.0, s: Float = 0.008)
```

Initialize the CosineNoiseSchedule.

<a id="mocoschedulesnoisecontinuous_snr_transformsLinearSNRTransform"></a>

## LinearSNRTransform Objects

```python
class LinearSNRTransform(ContinuousSNRTransform)
```

A Linear SNR schedule.

<a id="mocoschedulesnoisecontinuous_snr_transformsLinearSNRTransform__init__"></a>

#### \_\_init\_\_

```python
def __init__(min_value: Float = 1.0e-4)
```

Initialize the Linear SNR Transform.

**Arguments**:

- `min_value` _Float_ - min vaue of SNR defaults to 1.e-4.

<a id="mocoschedulesnoisecontinuous_snr_transformsLinearLogInterpolatedSNRTransform"></a>

## LinearLogInterpolatedSNRTransform Objects

```python
class LinearLogInterpolatedSNRTransform(ContinuousSNRTransform)
```

A Linear Log space interpolated SNR schedule.

<a id="mocoschedulesnoisecontinuous_snr_transformsLinearLogInterpolatedSNRTransform__init__"></a>

#### \_\_init\_\_

```python
def __init__(min_value: Float = -7.0, max_value=13.5)
```

Initialize the Linear log space interpolated SNR Schedule from Chroma.

**Arguments**:

- `min_value` _Float_ - The min log SNR value.
- `max_value` _Float_ - the max log SNR value.

<a id="mocoschedulesnoisediscrete_noise_schedules"></a>

# bionemo.moco.schedules.noise.discrete\_noise\_schedules

<a id="mocoschedulesnoisediscrete_noise_schedulesDiscreteNoiseSchedule"></a>

## DiscreteNoiseSchedule Objects

```python
class DiscreteNoiseSchedule(ABC)
```

A base class for discrete noise schedules.

<a id="mocoschedulesnoisediscrete_noise_schedulesDiscreteNoiseSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int, direction: TimeDirection)
```

Initialize the DiscreteNoiseSchedule.

**Arguments**:

- `nsteps` _int_ - number of discrete steps.
- `direction` _TimeDirection_ - required this defines in which direction the scheduler was built

<a id="mocoschedulesnoisediscrete_noise_schedulesDiscreteNoiseSchedulegenerate_schedule"></a>

#### generate\_schedule

```python
def generate_schedule(nsteps: Optional[int] = None,
                      device: Union[str, torch.device] = "cpu",
                      synchronize: Optional[TimeDirection] = None) -> Tensor
```

Generate the noise schedule as a tensor.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None, uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").
- `synchronize` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
  this parameter allows to flip the direction to match the specified one (default is None).

<a id="mocoschedulesnoisediscrete_noise_schedulesDiscreteNoiseSchedulecalculate_derivative"></a>

#### calculate\_derivative

```python
def calculate_derivative(
        nsteps: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None) -> Tensor
```

Calculate the time derivative of the schedule.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None, uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").
- `synchronize` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
  this parameter allows to flip the direction to match the specified one (default is None).


**Returns**:

- `Tensor` - A tensor representing the time derivative of the schedule.


**Raises**:

- `NotImplementedError` - If the derivative calculation is not implemented for this schedule.

<a id="mocoschedulesnoisediscrete_noise_schedulesDiscreteCosineNoiseSchedule"></a>

## DiscreteCosineNoiseSchedule Objects

```python
class DiscreteCosineNoiseSchedule(DiscreteNoiseSchedule)
```

A cosine discrete noise schedule.

<a id="mocoschedulesnoisediscrete_noise_schedulesDiscreteCosineNoiseSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int, nu: Float = 1.0, s: Float = 0.008)
```

Initialize the CosineNoiseSchedule.

**Arguments**:

- `nsteps` _int_ - Number of discrete steps.
- `nu` _Optional[Float]_ - Hyperparameter for the cosine schedule exponent (default is 1.0).
- `s` _Optional[Float]_ - Hyperparameter for the cosine schedule shift (default is 0.008).

<a id="mocoschedulesnoisediscrete_noise_schedulesDiscreteLinearNoiseSchedule"></a>

## DiscreteLinearNoiseSchedule Objects

```python
class DiscreteLinearNoiseSchedule(DiscreteNoiseSchedule)
```

A linear discrete noise schedule.

<a id="mocoschedulesnoisediscrete_noise_schedulesDiscreteLinearNoiseSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int, beta_start: Float = 1e-4, beta_end: Float = 0.02)
```

Initialize the CosineNoiseSchedule.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None, uses the value from initialization.
- `beta_start` _Optional[int]_ - starting beta value. Defaults to 1e-4.
- `beta_end` _Optional[int]_ - end beta value. Defaults to 0.02.

<a id="mocoschedulesnoise"></a>

# bionemo.moco.schedules.noise

<a id="mocoschedulesnoisecontinuous_noise_transforms"></a>

# bionemo.moco.schedules.noise.continuous\_noise\_transforms

<a id="mocoschedulesnoisecontinuous_noise_transformsContinuousExpNoiseTransform"></a>

## ContinuousExpNoiseTransform Objects

```python
class ContinuousExpNoiseTransform(ABC)
```

A base class for continuous schedules.

alpha = exp(- sigma) where 1 - alpha controls the masking fraction.

<a id="mocoschedulesnoisecontinuous_noise_transformsContinuousExpNoiseTransform__init__"></a>

#### \_\_init\_\_

```python
def __init__(direction: TimeDirection)
```

Initialize the DiscreteNoiseSchedule.

**Arguments**:

  direction : TimeDirection, required this defines in which direction the scheduler was built

<a id="mocoschedulesnoisecontinuous_noise_transformsContinuousExpNoiseTransformcalculate_sigma"></a>

#### calculate\_sigma

```python
def calculate_sigma(t: Tensor,
                    device: Union[str, torch.device] = "cpu",
                    synchronize: Optional[TimeDirection] = None) -> Tensor
```

Calculate the sigma for the given time steps.

**Arguments**:

- `t` _Tensor_ - The input tensor representing the time steps, with values ranging from 0 to 1.
- `device` _Optional[str]_ - The device to place the schedule on. Defaults to "cpu".
- `synchronize` _optional[TimeDirection]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
  this parameter allows to flip the direction to match the specified one. Defaults to None.


**Returns**:

- `Tensor` - A tensor representing the sigma values for the given time steps.


**Raises**:

- `ValueError` - If the input time steps exceed the maximum allowed value of 1.

<a id="mocoschedulesnoisecontinuous_noise_transformsContinuousExpNoiseTransformsigma_to_alpha"></a>

#### sigma\_to\_alpha

```python
def sigma_to_alpha(sigma: Tensor) -> Tensor
```

Converts sigma to alpha values by alpha = exp(- sigma).

**Arguments**:

- `sigma` _Tensor_ - The input sigma tensor.


**Returns**:

- `Tensor` - A tensor containing the alpha values.

<a id="mocoschedulesnoisecontinuous_noise_transformsCosineExpNoiseTransform"></a>

## CosineExpNoiseTransform Objects

```python
class CosineExpNoiseTransform(ContinuousExpNoiseTransform)
```

A cosine Exponential noise schedule.

<a id="mocoschedulesnoisecontinuous_noise_transformsCosineExpNoiseTransform__init__"></a>

#### \_\_init\_\_

```python
def __init__(eps: Float = 1.0e-3)
```

Initialize the CosineNoiseSchedule.

**Arguments**:

- `eps` _Float_ - small number to prevent numerical issues.

<a id="mocoschedulesnoisecontinuous_noise_transformsCosineExpNoiseTransformd_dt_sigma"></a>

#### d\_dt\_sigma

```python
def d_dt_sigma(t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor
```

Compute the derivative of sigma with respect to time.

**Arguments**:

- `t` _Tensor_ - The input tensor representing the time steps.
- `device` _Optional[str]_ - The device to place the schedule on. Defaults to "cpu".


**Returns**:

- `Tensor` - A tensor representing the derivative of sigma with respect to time.


**Notes**:

  The derivative of sigma as a function of time is given by:

  d/dt sigma(t) = d/dt (-log(cos(t * pi / 2) + eps))

  Using the chain rule, we get:

  d/dt sigma(t) = (-1 / (cos(t * pi / 2) + eps)) * (-sin(t * pi / 2) * pi / 2)

  This is the derivative that is computed and returned by this method.

<a id="mocoschedulesnoisecontinuous_noise_transformsLogLinearExpNoiseTransform"></a>

## LogLinearExpNoiseTransform Objects

```python
class LogLinearExpNoiseTransform(ContinuousExpNoiseTransform)
```

A log linear exponential schedule.

<a id="mocoschedulesnoisecontinuous_noise_transformsLogLinearExpNoiseTransform__init__"></a>

#### \_\_init\_\_

```python
def __init__(eps: Float = 1.0e-3)
```

Initialize the CosineNoiseSchedule.

**Arguments**:

- `eps` _Float_ - small value to prevent numerical issues.

<a id="mocoschedulesnoisecontinuous_noise_transformsLogLinearExpNoiseTransformd_dt_sigma"></a>

#### d\_dt\_sigma

```python
def d_dt_sigma(t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor
```

Compute the derivative of sigma with respect to time.

**Arguments**:

- `t` _Tensor_ - The input tensor representing the time steps.
- `device` _Optional[str]_ - The device to place the schedule on. Defaults to "cpu".


**Returns**:

- `Tensor` - A tensor representing the derivative of sigma with respect to time.

<a id="mocoschedules"></a>

# bionemo.moco.schedules

<a id="mocoschedulesutils"></a>

# bionemo.moco.schedules.utils

<a id="mocoschedulesutilsTimeDirection"></a>

## TimeDirection Objects

```python
class TimeDirection(Enum)
```

Enum for the direction of the noise schedule.

<a id="mocoschedulesutilsTimeDirectionUNIFIED"></a>

#### UNIFIED

Noise(0) --> Data(1)

<a id="mocoschedulesutilsTimeDirectionDIFFUSION"></a>

#### DIFFUSION

Noise(1) --> Data(0)

<a id="mocoschedulesinference_time_schedules"></a>

# bionemo.moco.schedules.inference\_time\_schedules

<a id="mocoschedulesinference_time_schedulesInferenceSchedule"></a>

## InferenceSchedule Objects

```python
class InferenceSchedule(ABC)
```

A base class for inference time schedules.

<a id="mocoschedulesinference_time_schedulesInferenceSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int,
             min_t: Float = 0,
             padding: Float = 0,
             dilation: Float = 0,
             direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
             device: Union[str, torch.device] = "cpu")
```

Initialize the InferenceSchedule.

**Arguments**:

- `nsteps` _int_ - Number of time steps.
- `min_t` _Float_ - minimum time value defaults to 0.
- `padding` _Float_ - padding time value defaults to 0.
- `dilation` _Float_ - dilation time value defaults to 0 ie the number of replicates.
- `direction` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocoschedulesinference_time_schedulesInferenceSchedulegenerate_schedule"></a>

#### generate\_schedule

```python
@abstractmethod
def generate_schedule(
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None) -> Tensor
```

Generate the time schedule as a tensor.

**Arguments**:

- `nsteps` _Optioanl[int]_ - Number of time steps. If None, uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocoschedulesinference_time_schedulesInferenceSchedulepad_time"></a>

#### pad\_time

```python
def pad_time(n_samples: int,
             scalar_time: Float,
             device: Optional[Union[str, torch.device]] = None) -> Tensor
```

Creates a tensor of shape (n_samples,) filled with a scalar time value.

**Arguments**:

- `n_samples` _int_ - The desired dimension of the output tensor.
- `scalar_time` _Float_ - The scalar time value to fill the tensor with.
  device (Optional[Union[str, torch.device]], optional):
  The device to place the tensor on. Defaults to None, which uses the default device.


**Returns**:

- `Tensor` - A tensor of shape (n_samples,) filled with the scalar time value.

<a id="mocoschedulesinference_time_schedulesContinuousInferenceSchedule"></a>

## ContinuousInferenceSchedule Objects

```python
class ContinuousInferenceSchedule(InferenceSchedule)
```

A base class for continuous time inference schedules.

<a id="mocoschedulesinference_time_schedulesContinuousInferenceSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int,
             inclusive_end: bool = False,
             min_t: Float = 0,
             padding: Float = 0,
             dilation: Float = 0,
             direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
             device: Union[str, torch.device] = "cpu")
```

Initialize the ContinuousInferenceSchedule.

**Arguments**:

- `nsteps` _int_ - Number of time steps.
- `inclusive_end` _bool_ - If True, include the end value (1.0) in the schedule otherwise ends at 1.0-1/nsteps (default is False).
- `min_t` _Float_ - minimum time value defaults to 0.
- `padding` _Float_ - padding time value defaults to 0.
- `dilation` _Float_ - dilation time value defaults to 0 ie the number of replicates.
- `direction` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocoschedulesinference_time_schedulesContinuousInferenceSchedulediscretize"></a>

#### discretize

```python
def discretize(nsteps: Optional[int] = None,
               schedule: Optional[Tensor] = None,
               device: Optional[Union[str, torch.device]] = None) -> Tensor
```

Discretize the time schedule into a list of time deltas.

**Arguments**:

- `nsteps` _Optioanl[int]_ - Number of time steps. If None, uses the value from initialization.
- `schedule` _Optional[Tensor]_ - Time scheudle if None will generate it with generate_schedule.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").


**Returns**:

- `Tensor` - A tensor of time deltas.

<a id="mocoschedulesinference_time_schedulesDiscreteInferenceSchedule"></a>

## DiscreteInferenceSchedule Objects

```python
class DiscreteInferenceSchedule(InferenceSchedule)
```

A base class for discrete time inference schedules.

<a id="mocoschedulesinference_time_schedulesDiscreteInferenceSchedulediscretize"></a>

#### discretize

```python
def discretize(nsteps: Optional[int] = None,
               device: Optional[Union[str, torch.device]] = None) -> Tensor
```

Discretize the time schedule into a list of time deltas.

**Arguments**:

- `nsteps` _Optioanl[int]_ - Number of time steps. If None, uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").


**Returns**:

- `Tensor` - A tensor of time deltas.

<a id="mocoschedulesinference_time_schedulesDiscreteLinearInferenceSchedule"></a>

## DiscreteLinearInferenceSchedule Objects

```python
class DiscreteLinearInferenceSchedule(DiscreteInferenceSchedule)
```

A linear time schedule for discrete time inference.

<a id="mocoschedulesinference_time_schedulesDiscreteLinearInferenceSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int,
             min_t: Float = 0,
             padding: Float = 0,
             dilation: Float = 0,
             direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
             device: Union[str, torch.device] = "cpu")
```

Initialize the DiscreteLinearInferenceSchedule.

**Arguments**:

- `nsteps` _int_ - Number of time steps.
- `min_t` _Float_ - minimum time value defaults to 0.
- `padding` _Float_ - padding time value defaults to 0.
- `dilation` _Float_ - dilation time value defaults to 0 ie the number of replicates.
- `direction` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocoschedulesinference_time_schedulesDiscreteLinearInferenceSchedulegenerate_schedule"></a>

#### generate\_schedule

```python
def generate_schedule(
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None) -> Tensor
```

Generate the linear time schedule as a tensor.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").


**Returns**:

- `Tensor` - A tensor of time steps.
- `Tensor` - A tensor of time steps.

<a id="mocoschedulesinference_time_schedulesLinearInferenceSchedule"></a>

## LinearInferenceSchedule Objects

```python
class LinearInferenceSchedule(ContinuousInferenceSchedule)
```

A linear time schedule for continuous time inference.

<a id="mocoschedulesinference_time_schedulesLinearInferenceSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int,
             inclusive_end: bool = False,
             min_t: Float = 0,
             padding: Float = 0,
             dilation: Float = 0,
             direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
             device: Union[str, torch.device] = "cpu")
```

Initialize the LinearInferenceSchedule.

**Arguments**:

- `nsteps` _int_ - Number of time steps.
- `inclusive_end` _bool_ - If True, include the end value (1.0) in the schedule otherwise ends at 1.0-1/nsteps (default is False).
- `min_t` _Float_ - minimum time value defaults to 0.
- `padding` _Float_ - padding time value defaults to 0.
- `dilation` _Float_ - dilation time value defaults to 0 ie the number of replicates.
- `direction` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocoschedulesinference_time_schedulesLinearInferenceSchedulegenerate_schedule"></a>

#### generate\_schedule

```python
def generate_schedule(
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None) -> Tensor
```

Generate the linear time schedule as a tensor.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").


**Returns**:

- `Tensor` - A tensor of time steps.

<a id="mocoschedulesinference_time_schedulesPowerInferenceSchedule"></a>

## PowerInferenceSchedule Objects

```python
class PowerInferenceSchedule(ContinuousInferenceSchedule)
```

A power time schedule for inference, where time steps are generated by raising a uniform schedule to a specified power.

<a id="mocoschedulesinference_time_schedulesPowerInferenceSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int,
             inclusive_end: bool = False,
             min_t: Float = 0,
             padding: Float = 0,
             dilation: Float = 0,
             exponent: Float = 1.0,
             direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
             device: Union[str, torch.device] = "cpu")
```

Initialize the PowerInferenceSchedule.

**Arguments**:

- `nsteps` _int_ - Number of time steps.
- `inclusive_end` _bool_ - If True, include the end value (1.0) in the schedule otherwise ends at <1.0 (default is False).
- `min_t` _Float_ - minimum time value defaults to 0.
- `padding` _Float_ - padding time value defaults to 0.
- `dilation` _Float_ - dilation time value defaults to 0 ie the number of replicates.
- `exponent` _Float_ - Power parameter defaults to 1.0.
- `direction` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocoschedulesinference_time_schedulesPowerInferenceSchedulegenerate_schedule"></a>

#### generate\_schedule

```python
def generate_schedule(
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None) -> Tensor
```

Generate the power time schedule as a tensor.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").



**Returns**:

- `Tensor` - A tensor of time steps.
- `Tensor` - A tensor of time steps.

<a id="mocoschedulesinference_time_schedulesLogInferenceSchedule"></a>

## LogInferenceSchedule Objects

```python
class LogInferenceSchedule(ContinuousInferenceSchedule)
```

A log time schedule for inference, where time steps are generated by taking the logarithm of a uniform schedule.

<a id="mocoschedulesinference_time_schedulesLogInferenceSchedule__init__"></a>

#### \_\_init\_\_

```python
def __init__(nsteps: int,
             inclusive_end: bool = False,
             min_t: Float = 0,
             padding: Float = 0,
             dilation: Float = 0,
             exponent: Float = -2.0,
             direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
             device: Union[str, torch.device] = "cpu")
```

Initialize the LogInferenceSchedule.

Returns a log space time schedule.

Which for 100 steps with default parameters is:
tensor([0.0000, 0.0455, 0.0889, 0.1303, 0.1699, 0.2077, 0.2439, 0.2783, 0.3113,
0.3427, 0.3728, 0.4015, 0.4288, 0.4550, 0.4800, 0.5039, 0.5266, 0.5484,
0.5692, 0.5890, 0.6080, 0.6261, 0.6434, 0.6599, 0.6756, 0.6907, 0.7051,
0.7188, 0.7319, 0.7444, 0.7564, 0.7678, 0.7787, 0.7891, 0.7991, 0.8086,
0.8176, 0.8263, 0.8346, 0.8425, 0.8500, 0.8572, 0.8641, 0.8707, 0.8769,
0.8829, 0.8887, 0.8941, 0.8993, 0.9043, 0.9091, 0.9136, 0.9180, 0.9221,
0.9261, 0.9299, 0.9335, 0.9369, 0.9402, 0.9434, 0.9464, 0.9492, 0.9520,
0.9546, 0.9571, 0.9595, 0.9618, 0.9639, 0.9660, 0.9680, 0.9699, 0.9717,
0.9734, 0.9751, 0.9767, 0.9782, 0.9796, 0.9810, 0.9823, 0.9835, 0.9847,
0.9859, 0.9870, 0.9880, 0.9890, 0.9899, 0.9909, 0.9917, 0.9925, 0.9933,
0.9941, 0.9948, 0.9955, 0.9962, 0.9968, 0.9974, 0.9980, 0.9985, 0.9990,
0.9995])

**Arguments**:

- `nsteps` _int_ - Number of time steps.
- `inclusive_end` _bool_ - If True, include the end value (1.0) in the schedule otherwise ends at <1.0 (default is False).
- `min_t` _Float_ - minimum time value defaults to 0.
- `padding` _Float_ - padding time value defaults to 0.
- `dilation` _Float_ - dilation time value defaults to 0 ie the number of replicates.
- `exponent` _Float_ - log space exponent parameter defaults to -2.0. The lower number the more aggressive the acceleration of 0 to 0.9 will be thus having more steps from 0.9 to 1.0.
- `direction` _Optional[str]_ - TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocoschedulesinference_time_schedulesLogInferenceSchedulegenerate_schedule"></a>

#### generate\_schedule

```python
def generate_schedule(
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None) -> Tensor
```

Generate the log time schedule as a tensor.

**Arguments**:

- `nsteps` _Optional[int]_ - Number of time steps. If None uses the value from initialization.
- `device` _Optional[str]_ - Device to place the schedule on (default is "cpu").

<a id="mocointerpolantscontinuous_timediscrete"></a>

# bionemo.moco.interpolants.continuous\_time.discrete

<a id="mocointerpolantscontinuous_timediscretemdlm"></a>

# bionemo.moco.interpolants.continuous\_time.discrete.mdlm

<a id="mocointerpolantscontinuous_timediscretemdlmMDLM"></a>

## MDLM Objects

```python
class MDLM(Interpolant)
```

A Masked discrete Diffusion Language Model (MDLM) interpolant.

-------

**Examples**:

```python
>>> import torch
>>> from bionemo.bionemo.moco.distributions.prior.discrete.mask import DiscreteMaskedPrior
>>> from bionemo.bionemo.moco.distributions.time.uniform import UniformTimeDistribution
>>> from bionemo.bionemo.moco.interpolants.continuous_time.discrete.mdlm import MDLM
>>> from bionemo.bionemo.moco.schedules.noise.continuous_noise_transforms import CosineExpNoiseTransform
>>> from bionemo.bionemo.moco.schedules.inference_time_schedules import LinearTimeSchedule


mdlm = MDLM(
    time_distribution = UniformTimeDistribution(discrete_time = False,...),
    prior_distribution = DiscreteMaskedPrior(...),
    noise_schedule = CosineExpNoiseTransform(...),
    )
model = Model(...)

# Training
for epoch in range(1000):
    data = data_loader.get(...)
    time = mdlm.sample_time(batch_size)
    xt = mdlm.interpolate(data, time)

    logits = model(xt, time)
    loss = mdlm.loss(logits, data, xt, time)
    loss.backward()

# Generation
x_pred = mdlm.sample_prior(data.shape)
schedule = LinearTimeSchedule(...)
inference_time = schedule.generate_schedule()
dts = schedue.discreteize()
for t, dt in zip(inference_time, dts):
    time = torch.full((batch_size,), t)
    logits = model(x_pred, time)
    x_pred = mdlm.step(logits, time, x_pred, dt)
return x_pred

```

<a id="mocointerpolantscontinuous_timediscretemdlmMDLM__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_distribution: TimeDistribution,
             prior_distribution: DiscreteMaskedPrior,
             noise_schedule: ContinuousExpNoiseTransform,
             device: str = "cpu",
             rng_generator: Optional[torch.Generator] = None)
```

Initialize the Masked Discrete Language Model (MDLM) interpolant.

**Arguments**:

- `time_distribution` _TimeDistribution_ - The distribution governing the time variable in the diffusion process.
- `prior_distribution` _DiscreteMaskedPrior_ - The prior distribution over the discrete token space, including masked tokens.
- `noise_schedule` _ContinuousExpNoiseTransform_ - The noise schedule defining the noise intensity as a function of time.
- `device` _str, optional_ - The device to use for computations. Defaults to "cpu".
- `rng_generator` _Optional[torch.Generator], optional_ - The random number generator for reproducibility. Defaults to None.

<a id="mocointerpolantscontinuous_timediscretemdlmMDLMinterpolate"></a>

#### interpolate

```python
def interpolate(data: Tensor, t: Tensor)
```

Get x(t) with given time t from noise and data.

**Arguments**:

- `data` _Tensor_ - target discrete ids
- `t` _Tensor_ - time

<a id="mocointerpolantscontinuous_timediscretemdlmMDLMforward_process"></a>

#### forward\_process

```python
def forward_process(data: Tensor, t: Tensor) -> Tensor
```

Apply the forward process to the data at time t.

**Arguments**:

- `data` _Tensor_ - target discrete ids
- `t` _Tensor_ - time


**Returns**:

- `Tensor` - x(t) after applying the forward process

<a id="mocointerpolantscontinuous_timediscretemdlmMDLMloss"></a>

#### loss

```python
def loss(logits: Tensor,
         target: Tensor,
         xt: Tensor,
         time: Tensor,
         mask: Optional[Tensor] = None,
         use_weight=True)
```

Calculate the cross-entropy loss between the model prediction and the target output.

The loss is calculated between the batch x node x class logits and the target batch x node,
considering the current state of the discrete sequence `xt` at time `time`.

If `use_weight` is True, the loss is weighted by the reduced form of the MDLM time weight for continuous NELBO,
as specified in equation 11 of https://arxiv.org/pdf/2406.07524. This weight is proportional to the derivative
of the noise schedule with respect to time, and is used to emphasize the importance of accurate predictions at
certain times in the diffusion process.

**Arguments**:

- `logits` _Tensor_ - The predicted output from the model, with shape batch x node x class.
- `target` _Tensor_ - The target output for the model prediction, with shape batch x node.
- `xt` _Tensor_ - The current state of the discrete sequence, with shape batch x node.
- `time` _Tensor_ - The time at which the loss is calculated.
- `mask` _Optional[Tensor], optional_ - The mask for the data point. Defaults to None.
- `use_weight` _bool, optional_ - Whether to use the MDLM time weight for the loss. Defaults to True.


**Returns**:

- `Tensor` - The calculated loss batch tensor.

<a id="mocointerpolantscontinuous_timediscretemdlmMDLMstep"></a>

#### step

```python
def step(logits, t, xt, dt) -> Tensor
```

Perform a single step of MDLM DDPM step.

**Arguments**:

- `logits` _Tensor_ - The input logits.
- `t` _float_ - The current time step.
- `xt` _Tensor_ - The current state.
- `dt` _float_ - The time step increment.


**Returns**:

- `Tensor` - The updated state.

<a id="mocointerpolantscontinuous_timediscretemdlmMDLMstep_confidence"></a>

#### step\_confidence

```python
def step_confidence(logits: Tensor,
                    xt: Tensor,
                    curr_step: int,
                    num_steps: int,
                    logit_temperature: float = 1.0,
                    randomness: float = 1.0,
                    confidence_temperature: float = 1.0) -> Tensor
```

Update the input sequence xt by sampling from the predicted logits and adding Gumbel noise.

Method taken from GenMol Seul et al.

**Arguments**:

- `logits` - Predicted logits
- `xt` - Input sequence
- `curr_step` - Current step
- `num_steps` - Total number of steps
- `logit_temperature` - Temperature for softmax over logits
- `randomness` - Scale for Gumbel noise
- `confidence_temperature` - Temperature for Gumbel confidence


**Returns**:

  Updated input sequence xt

<a id="mocointerpolantscontinuous_timediscretemdlmMDLMstep_argmax"></a>

#### step\_argmax

```python
def step_argmax(model_out: Tensor)
```

Returns the index of the maximum value in the last dimension of the model output.

**Arguments**:

- `model_out` _Tensor_ - The output of the model.


**Returns**:

- `Tensor` - The index of the maximum value in the last dimension of the model output.

<a id="mocointerpolantscontinuous_timediscretemdlmMDLMcalculate_score"></a>

#### calculate\_score

```python
def calculate_score(logits, x, t)
```

Returns score of the given sample x at time t with the corresponding model output logits.

**Arguments**:

- `logits` _Tensor_ - The output of the model.
- `x` _Tensor_ - The current data point.
- `t` _Tensor_ - The current time.


**Returns**:

- `Tensor` - The score defined in Appendix C.3 Equation 76 of MDLM.

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matching"></a>

# bionemo.moco.interpolants.continuous\_time.discrete.discrete\_flow\_matching

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matchingDiscreteFlowMatcher"></a>

## DiscreteFlowMatcher Objects

```python
class DiscreteFlowMatcher(Interpolant)
```

A Discrete Flow Model (DFM) interpolant.

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matchingDiscreteFlowMatcher__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_distribution: TimeDistribution,
             prior_distribution: DiscretePriorDistribution,
             device: str = "cpu",
             eps: Float = 1e-5,
             rng_generator: Optional[torch.Generator] = None)
```

Initialize the DFM interpolant.

**Arguments**:

- `time_distribution` _TimeDistribution_ - The time distribution for the diffusion process.
- `prior_distribution` _DiscretePriorDistribution_ - The prior distribution for the discrete masked tokens.
- `device` _str, optional_ - The device to use for computations. Defaults to "cpu".
- `eps` - small Float to prevent dividing by zero.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matchingDiscreteFlowMatcherinterpolate"></a>

#### interpolate

```python
def interpolate(data: Tensor, t: Tensor, noise: Tensor)
```

Get x(t) with given time t from noise and data.

**Arguments**:

- `data` _Tensor_ - target discrete ids
- `t` _Tensor_ - time
- `noise` - tensor noise ids

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matchingDiscreteFlowMatcherloss"></a>

#### loss

```python
def loss(logits: Tensor,
         target: Tensor,
         time: Optional[Tensor] = None,
         mask: Optional[Tensor] = None,
         use_weight: Bool = False)
```

Calculate the cross-entropy loss between the model prediction and the target output.

The loss is calculated between the batch x node x class logits and the target batch x node.
If using a masked prior please pass in the correct mask to calculate loss values on only masked states.
i.e. mask = data_mask * is_masked_state which is calculated with self.prior_dist.is_masked(xt))

If `use_weight` is True, the loss is weighted by 1/(1-t) defined in equation 24 in Appndix C. of https://arxiv.org/pdf/2402.04997

**Arguments**:

- `logits` _Tensor_ - The predicted output from the model, with shape batch x node x class.
- `target` _Tensor_ - The target output for the model prediction, with shape batch x node.
- `time` _Tensor_ - The time at which the loss is calculated.
- `mask` _Optional[Tensor], optional_ - The mask for the data point. Defaults to None.
- `use_weight` _bool, optional_ - Whether to use the DFM time weight for the loss. Defaults to True.


**Returns**:

- `Tensor` - The calculated loss batch tensor.

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matchingDiscreteFlowMatcherstep"></a>

#### step

```python
def step(logits: Tensor,
         t: Tensor,
         xt: Tensor,
         dt: Tensor | float,
         temperature: Float = 1.0,
         stochasticity: Float = 1.0) -> Tensor
```

Perform a single step of DFM euler updates.

**Arguments**:

- `logits` _Tensor_ - The input logits.
- `t` _Tensor_ - The current time step.
- `xt` _Tensor_ - The current state.
- `dt` _Tensor | float_ - The time step increment.
- `temperature` _Float, optional_ - The temperature for the softmax calculation. Defaults to 1.0.
- `stochasticity` _Float, optional_ - The stochasticity value for the step calculation. Defaults to 1.0.


**Returns**:

- `Tensor` - The updated state.

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matchingDiscreteFlowMatcherstep_purity"></a>

#### step\_purity

```python
def step_purity(logits: Tensor,
                t: Tensor,
                xt: Tensor,
                dt: Tensor | float,
                temperature: Float = 1.0,
                stochasticity: Float = 1.0) -> Tensor
```

Perform a single step of purity sampling.

https://github.com/jasonkyuyim/multiflow/blob/6278899970523bad29953047e7a42b32a41dc813/multiflow/data/interpolant.py#L346
Here's a high-level overview of what the function does:
TODO: check if the -1e9 and 1e-9 are small enough or using torch.inf would be better

1. Preprocessing:
Checks if dt is a float and converts it to a tensor if necessary.
Pads t and dt to match the shape of xt.
Checks if the mask_index is valid (i.e., within the range of possible discrete values).
2. Masking:
Sets the logits corresponding to the mask_index to a low value (-1e9) to effectively mask out those values.
Computes the softmax probabilities of the logits.
Sets the probability of the mask_index to a small value (1e-9) to avoid numerical issues.
3.Purity sampling:
Computes the maximum log probabilities of the softmax distribution.
Computes the indices of the top-number_to_unmask samples with the highest log probabilities.
Uses these indices to sample new values from the original distribution.
4. Unmasking and updating:
Creates a mask to select the top-number_to_unmask samples.
Uses this mask to update the current state xt with the new samples.
5. Re-masking:
Generates a new mask to randomly re-mask some of the updated samples.
Applies this mask to the updated state xt.

**Arguments**:

- `logits` _Tensor_ - The input logits.
- `t` _Tensor_ - The current time step.
- `xt` _Tensor_ - The current state.
- `dt` _Tensor_ - The time step increment.
- `temperature` _Float, optional_ - The temperature for the softmax calculation. Defaults to 1.0.
- `stochasticity` _Float, optional_ - The stochasticity value for the step calculation. Defaults to 1.0.


**Returns**:

- `Tensor` - The updated state.

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matchingDiscreteFlowMatcherstep_argmax"></a>

#### step\_argmax

```python
def step_argmax(model_out: Tensor)
```

Returns the index of the maximum value in the last dimension of the model output.

**Arguments**:

- `model_out` _Tensor_ - The output of the model.

<a id="mocointerpolantscontinuous_timediscretediscrete_flow_matchingDiscreteFlowMatcherstep_simple_sample"></a>

#### step\_simple\_sample

```python
def step_simple_sample(model_out: Tensor,
                       temperature: float = 1.0,
                       num_samples: int = 1)
```

Samples from the model output logits. Leads to more diversity than step_argmax.

**Arguments**:

- `model_out` _Tensor_ - The output of the model.
- `temperature` _Float, optional_ - The temperature for the softmax calculation. Defaults to 1.0.
- `num_samples` _int_ - Number of samples to return

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_types"></a>

# bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport.ot\_types

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_typesOptimalTransportType"></a>

## OptimalTransportType Objects

```python
class OptimalTransportType(Enum)
```

An enumeration representing the type ofOptimal Transport that can be used in Continuous Flow Matching.

- **EXACT**: Standard mini batch optimal transport defined in  https://arxiv.org/pdf/2302.00482.
- **EQUIVARIANT**: Adding roto/translation optimization to mini batch OT see https://arxiv.org/pdf/2306.15030  https://arxiv.org/pdf/2312.07168 4.2.
- **KABSCH**: Simple Kabsch alignment between each data and noise point, No permuation # https://arxiv.org/pdf/2410.22388 Sec 3.2

These prediction types can be used to train neural networks for specific tasks, such as denoising, image synthesis, or time-series forecasting.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_sampler"></a>

# bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport.ot\_sampler

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_samplerOTSampler"></a>

## OTSampler Objects

```python
class OTSampler()
```

Sampler for Exact Mini-batch Optimal Transport Plan.

OTSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean cost)
with different implementations of the plan calculation. Code is adapted from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_samplerOTSampler__init__"></a>

#### \_\_init\_\_

```python
def __init__(method: str = "exact",
             device: Union[str, torch.device] = "cpu",
             num_threads: int = 1) -> None
```

Initialize the OTSampler class.

**Arguments**:

- `method` _str_ - Choose which optimal transport solver you would like to use. Currently only support exact OT solvers (pot.emd).
- `device` _Union[str, torch.device], optional_ - The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
- `num_threads` _Union[int, str], optional_ - Number of threads to use for OT solver. If "max", uses the maximum number of threads. Default is 1.


**Raises**:

- `ValueError` - If the OT solver is not documented.
- `NotImplementedError` - If the OT solver is not implemented.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_samplerOTSamplerto_device"></a>

#### to\_device

```python
def to_device(device: str)
```

Moves all internal tensors to the specified device and updates the `self.device` attribute.

**Arguments**:

- `device` _str_ - The device to move the tensors to (e.g. "cpu", "cuda:0").


**Notes**:

  This method is used to transfer the internal state of the OTSampler to a different device.
  It updates the `self.device` attribute to reflect the new device and moves all internal tensors to the specified device.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_samplerOTSamplersample_map"></a>

#### sample\_map

```python
def sample_map(pi: Tensor,
               batch_size: int,
               replace: Bool = False) -> Tuple[Tensor, Tensor]
```

Draw source and target samples from pi $(x,z) \sim \pi$.

**Arguments**:

- `pi` _Tensor_ - shape (bs, bs), the OT matrix between noise and data in minibatch.
- `batch_size` _int_ - The batch size of the minibatch.
- `replace` _bool_ - sampling w/ or w/o replacement from the OT plan, default to False.


**Returns**:

- `Tuple` - tuple of 2 tensors, represents the indices of noise and data samples from pi.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_samplerOTSamplerget_ot_matrix"></a>

#### get\_ot\_matrix

```python
def get_ot_matrix(x0: Tensor,
                  x1: Tensor,
                  mask: Optional[Tensor] = None) -> Tensor
```

Compute the OT matrix between a source and a target minibatch.

**Arguments**:

- `x0` _Tensor_ - shape (bs, *dim), noise from source minibatch.
- `x1` _Tensor_ - shape (bs, *dim), data from source minibatch.
- `mask` _Optional[Tensor], optional_ - mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.


**Returns**:

- `p` _Tensor_ - shape (bs, bs), the OT matrix between noise and data in minibatch.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportot_samplerOTSamplerapply_ot"></a>

#### apply\_ot

```python
def apply_ot(
    x0: Tensor,
    x1: Tensor,
    mask: Optional[Tensor] = None,
    replace: Bool = False,
    sort: Optional[Literal["noise", "x0", "data", "x1"]] = "x0"
) -> Tuple[Tensor, Tensor, Optional[Tensor]]
```

Sample indices for noise and data in minibatch according to OT plan.

Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
minibatch and draw source and target samples from pi $(x,z) \sim \pi$.

**Arguments**:

- `x0` _Tensor_ - shape (bs, *dim), noise from source minibatch.
- `x1` _Tensor_ - shape (bs, *dim), data from source minibatch.
- `mask` _Optional[Tensor], optional_ - mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.
- `replace` _bool_ - sampling w/ or w/o replacement from the OT plan, default to False.
- `sort` _str_ - Optional Literal string to sort either x1 or x0 based on the input.


**Returns**:

- `Tuple` - tuple of 2 tensors or 3 tensors if mask is used, represents the noise (plus mask) and data samples following OT plan pi.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_sampler"></a>

# bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport.equivariant\_ot\_sampler

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_samplerEquivariantOTSampler"></a>

## EquivariantOTSampler Objects

```python
class EquivariantOTSampler()
```

Sampler for Mini-batch Optimal Transport Plan with cost calculated after Kabsch alignment.

EquivariantOTSampler implements sampling coordinates according to an OT plan
(wrt squared Euclidean cost after Kabsch alignment) with different implementations of the plan calculation.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_samplerEquivariantOTSampler__init__"></a>

#### \_\_init\_\_

```python
def __init__(method: str = "exact",
             device: Union[str, torch.device] = "cpu",
             num_threads: int = 1) -> None
```

Initialize the OTSampler class.

**Arguments**:

- `method` _str_ - Choose which optimal transport solver you would like to use. Currently only support exact OT solvers (pot.emd).
- `device` _Union[str, torch.device], optional_ - The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
- `num_threads` _Union[int, str], optional_ - Number of threads to use for OT solver. If "max", uses the maximum number of threads. Default is 1.


**Raises**:

- `ValueError` - If the OT solver is not documented.
- `NotImplementedError` - If the OT solver is not implemented.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_samplerEquivariantOTSamplerto_device"></a>

#### to\_device

```python
def to_device(device: str)
```

Moves all internal tensors to the specified device and updates the `self.device` attribute.

**Arguments**:

- `device` _str_ - The device to move the tensors to (e.g. "cpu", "cuda:0").


**Notes**:

  This method is used to transfer the internal state of the OTSampler to a different device.
  It updates the `self.device` attribute to reflect the new device and moves all internal tensors to the specified device.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_samplerEquivariantOTSamplersample_map"></a>

#### sample\_map

```python
def sample_map(pi: Tensor,
               batch_size: int,
               replace: Bool = False) -> Tuple[Tensor, Tensor]
```

Draw source and target samples from pi $(x,z) \sim \pi$.

**Arguments**:

- `pi` _Tensor_ - shape (bs, bs), the OT matrix between noise and data in minibatch.
- `batch_size` _int_ - The batch size of the minibatch.
- `replace` _bool_ - sampling w/ or w/o replacement from the OT plan, default to False.


**Returns**:

- `Tuple` - tuple of 2 tensors, represents the indices of noise and data samples from pi.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_samplerEquivariantOTSamplerkabsch_align"></a>

#### kabsch\_align

```python
def kabsch_align(target: Tensor, noise: Tensor) -> Tensor
```

Find the Rotation matrix (R) such that RMSD is minimized between target @ R.T and noise.

**Arguments**:

- `target` _Tensor_ - shape (N, *dim), data from source minibatch.
- `noise` _Tensor_ - shape (N, *dim), noise from source minibatch.


**Returns**:

- `R` _Tensor_ - shape (*dim, *dim), the rotation matrix.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_samplerEquivariantOTSamplerget_ot_matrix"></a>

#### get\_ot\_matrix

```python
def get_ot_matrix(x0: Tensor,
                  x1: Tensor,
                  mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]
```

Compute the OT matrix between a source and a target minibatch.

**Arguments**:

- `x0` _Tensor_ - shape (bs, *dim), noise from source minibatch.
- `x1` _Tensor_ - shape (bs, *dim), data from source minibatch.
- `mask` _Optional[Tensor], optional_ - mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.


**Returns**:

- `p` _Tensor_ - shape (bs, bs), the OT matrix between noise and data in minibatch.
- `Rs` _Tensor_ - shape (bs, bs, *dim, *dim), the rotation matrix between noise and data in minibatch.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportequivariant_ot_samplerEquivariantOTSamplerapply_ot"></a>

#### apply\_ot

```python
def apply_ot(
    x0: Tensor,
    x1: Tensor,
    mask: Optional[Tensor] = None,
    replace: Bool = False,
    sort: Optional[Literal["noise", "x0", "data", "x1"]] = "x0"
) -> Tuple[Tensor, Tensor, Optional[Tensor]]
```

Sample indices for noise and data in minibatch according to OT plan.

Compute the OT plan $\pi$ (wrt squared Euclidean cost after Kabsch alignment) between a source and a target
minibatch and draw source and target samples from pi $(x,z) \sim \pi$.

**Arguments**:

- `x0` _Tensor_ - shape (bs, *dim), noise from source minibatch.
- `x1` _Tensor_ - shape (bs, *dim), data from source minibatch.
- `mask` _Optional[Tensor], optional_ - mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.
- `replace` _bool_ - sampling w/ or w/o replacement from the OT plan, default to False.
- `sort` _str_ - Optional Literal string to sort either x1 or x0 based on the input.


**Returns**:

- `Tuple` - tuple of 2 tensors, represents the noise and data samples following OT plan pi.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportkabsch_augmentation"></a>

# bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport.kabsch\_augmentation

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportkabsch_augmentationKabschAugmentation"></a>

## KabschAugmentation Objects

```python
class KabschAugmentation()
```

Point-wise Kabsch alignment.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportkabsch_augmentationKabschAugmentation__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

Initialize the KabschAugmentation instance.

**Notes**:

  - This implementation assumes no required initialization arguments.
  - You can add instance variables (e.g., `self.variable_name`) as needed.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportkabsch_augmentationKabschAugmentationkabsch_align"></a>

#### kabsch\_align

```python
def kabsch_align(target: Tensor, noise: Tensor)
```

Find the Rotation matrix (R) such that RMSD is minimized between target @ R.T and noise.

**Arguments**:

- `target` _Tensor_ - shape (N, *dim), data from source minibatch.
- `noise` _Tensor_ - shape (N, *dim), noise from source minibatch.


**Returns**:

- `R` _Tensor_ - shape (*dim, *dim), the rotation matrix.
  Aliged Target (Tensor): target tensor rotated and shifted to reduced RMSD with noise

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportkabsch_augmentationKabschAugmentationbatch_kabsch_align"></a>

#### batch\_kabsch\_align

```python
def batch_kabsch_align(target: Tensor, noise: Tensor)
```

Find the Rotation matrix (R) such that RMSD is minimized between target @ R.T and noise.

**Arguments**:

- `target` _Tensor_ - shape (N, *dim), data from source minibatch.
- `noise` _Tensor_ - shape (N, *dim), noise from source minibatch.


**Returns**:

- `R` _Tensor_ - shape (*dim, *dim), the rotation matrix.
  Aliged Target (Tensor): target tensor rotated and shifted to reduced RMSD with noise

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transportkabsch_augmentationKabschAugmentationapply_ot"></a>

#### apply\_ot

```python
def apply_ot(x0: Tensor,
             x1: Tensor,
             mask: Optional[Tensor] = None,
             align_noise_to_data=True) -> Tuple[Tensor, Tensor]
```

Sample indices for noise and data in minibatch according to OT plan.

Compute the OT plan $\pi$ (wrt squared Euclidean cost after Kabsch alignment) between a source and a target
minibatch and draw source and target samples from pi $(x,z) \sim \pi$.

**Arguments**:

- `x0` _Tensor_ - shape (bs, *dim), noise from source minibatch.
- `x1` _Tensor_ - shape (bs, *dim), data from source minibatch.
- `mask` _Optional[Tensor], optional_ - mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.
- `replace` _bool_ - sampling w/ or w/o replacement from the OT plan, default to False.
- `align_noise_to_data` _bool_ - Direction of alignment default is True meaning it augments Noise to reduce error to Data.


**Returns**:

- `Tuple` - tuple of 2 tensors, represents the noise and data samples following OT plan pi.

<a id="mocointerpolantscontinuous_timecontinuousoptimal_transport"></a>

# bionemo.moco.interpolants.continuous\_time.continuous.optimal\_transport

<a id="mocointerpolantscontinuous_timecontinuous"></a>

# bionemo.moco.interpolants.continuous\_time.continuous

<a id="mocointerpolantscontinuous_timecontinuousvdm"></a>

# bionemo.moco.interpolants.continuous\_time.continuous.vdm

<a id="mocointerpolantscontinuous_timecontinuousvdmVDM"></a>

## VDM Objects

```python
class VDM(Interpolant)
```

A Variational Diffusion Models (VDM) interpolant.

-------

**Examples**:

```python
>>> import torch
>>> from bionemo.bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
>>> from bionemo.bionemo.moco.distributions.time.uniform import UniformTimeDistribution
>>> from bionemo.bionemo.moco.interpolants.discrete_time.continuous.vdm import VDM
>>> from bionemo.bionemo.moco.schedules.noise.continuous_snr_transforms import CosineSNRTransform
>>> from bionemo.bionemo.moco.schedules.inference_time_schedules import LinearInferenceSchedule


vdm = VDM(
    time_distribution = UniformTimeDistribution(...),
    prior_distribution = GaussianPrior(...),
    noise_schedule = CosineSNRTransform(...),
    )
model = Model(...)

# Training
for epoch in range(1000):
    data = data_loader.get(...)
    time = vdm.sample_time(batch_size)
    noise = vdm.sample_prior(data.shape)
    xt = vdm.interpolate(data, noise, time)

    x_pred = model(xt, time)
    loss = vdm.loss(x_pred, data, time)
    loss.backward()

# Generation
x_pred = vdm.sample_prior(data.shape)
for t in LinearInferenceSchedule(...).generate_schedule():
    time = torch.full((batch_size,), t)
    x_hat = model(x_pred, time)
    x_pred = vdm.step(x_hat, time, x_pred)
return x_pred

```

<a id="mocointerpolantscontinuous_timecontinuousvdmVDM__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_distribution: TimeDistribution,
             prior_distribution: PriorDistribution,
             noise_schedule: ContinuousSNRTransform,
             prediction_type: Union[PredictionType, str] = PredictionType.DATA,
             device: Union[str, torch.device] = "cpu",
             rng_generator: Optional[torch.Generator] = None)
```

Initializes the DDPM interpolant.

**Arguments**:

- `time_distribution` _TimeDistribution_ - The distribution of time steps, used to sample time points for the diffusion process.
- `prior_distribution` _PriorDistribution_ - The prior distribution of the variable, used as the starting point for the diffusion process.
- `noise_schedule` _ContinuousSNRTransform_ - The schedule of noise, defining the amount of noise added at each time step.
- `prediction_type` _PredictionType, optional_ - The type of prediction, either "data" or another type. Defaults to "data".
- `device` _str, optional_ - The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMinterpolate"></a>

#### interpolate

```python
def interpolate(data: Tensor, t: Tensor, noise: Tensor)
```

Get x(t) with given time t from noise and data.

**Arguments**:

- `data` _Tensor_ - target
- `t` _Tensor_ - time
- `noise` _Tensor_ - noise from prior()

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMforward_process"></a>

#### forward\_process

```python
def forward_process(data: Tensor, t: Tensor, noise: Optional[Tensor] = None)
```

Get x(t) with given time t from noise and data.

**Arguments**:

- `data` _Tensor_ - target
- `t` _Tensor_ - time
- `noise` _Tensor, optional_ - noise from prior(). Defaults to None

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMprocess_data_prediction"></a>

#### process\_data\_prediction

```python
def process_data_prediction(model_output: Tensor, sample, t)
```

Converts the model output to a data prediction based on the prediction type.

This conversion stems from the Progressive Distillation for Fast Sampling of Diffusion Models https://arxiv.org/pdf/2202.00512.
Given the model output and the sample, we convert the output to a data prediction based on the prediction type.
The conversion formulas are as follows:
- For "noise" prediction type: `pred_data = (sample - noise_scale * model_output) / data_scale`
- For "data" prediction type: `pred_data = model_output`
- For "v_prediction" prediction type: `pred_data = data_scale * sample - noise_scale * model_output`

**Arguments**:

- `model_output` _Tensor_ - The output of the model.
- `sample` _Tensor_ - The input sample.
- `t` _Tensor_ - The time step.


**Returns**:

  The data prediction based on the prediction type.


**Raises**:

- `ValueError` - If the prediction type is not one of "noise", "data", or "v_prediction".

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMprocess_noise_prediction"></a>

#### process\_noise\_prediction

```python
def process_noise_prediction(model_output: Tensor, sample: Tensor, t: Tensor)
```

Do the same as process_data_prediction but take the model output and convert to nosie.

**Arguments**:

- `model_output` _Tensor_ - The output of the model.
- `sample` _Tensor_ - The input sample.
- `t` _Tensor_ - The time step.


**Returns**:

  The input as noise if the prediction type is "noise".


**Raises**:

- `ValueError` - If the prediction type is not "noise".

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMstep"></a>

#### step

```python
def step(model_out: Tensor,
         t: Tensor,
         xt: Tensor,
         dt: Tensor,
         mask: Optional[Tensor] = None,
         center: Bool = False,
         temperature: Float = 1.0)
```

Do one step integration.

**Arguments**:

- `model_out` _Tensor_ - The output of the model.
- `xt` _Tensor_ - The current data point.
- `t` _Tensor_ - The current time step.
- `dt` _Tensor_ - The time step increment.
- `mask` _Optional[Tensor], optional_ - An optional mask to apply to the data. Defaults to None.
- `center` _bool_ - Whether to center the data. Defaults to False.
- `temperature` _Float_ - The temperature parameter for low temperature sampling. Defaults to 1.0.


**Notes**:

  The temperature parameter controls the trade off between diversity and sample quality.
  Decreasing the temperature sharpens the sampling distribtion to focus on more likely samples.
  The impact of low temperature sampling must be ablated analytically.

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMscore"></a>

#### score

```python
def score(x_hat: Tensor, xt: Tensor, t: Tensor)
```

Converts the data prediction to the estimated score function.

**Arguments**:

- `x_hat` _tensor_ - The predicted data point.
- `xt` _Tensor_ - The current data point.
- `t` _Tensor_ - The time step.


**Returns**:

  The estimated score function.

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMstep_ddim"></a>

#### step\_ddim

```python
def step_ddim(model_out: Tensor,
              t: Tensor,
              xt: Tensor,
              dt: Tensor,
              mask: Optional[Tensor] = None,
              eta: Float = 0.0,
              center: Bool = False)
```

Do one step of DDIM sampling.

From the ddpm equations alpha_bar = alpha**2 and  1 - alpha**2 = sigma**2

**Arguments**:

- `model_out` _Tensor_ - output of the model
- `t` _Tensor_ - current time step
- `xt` _Tensor_ - current data point
- `dt` _Tensor_ - The time step increment.
- `mask` _Optional[Tensor], optional_ - mask for the data point. Defaults to None.
- `eta` _Float, optional_ - DDIM sampling parameter. Defaults to 0.0.
- `center` _Bool, optional_ - whether to center the data point. Defaults to False.

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMset_loss_weight_fn"></a>

#### set\_loss\_weight\_fn

```python
def set_loss_weight_fn(fn: Callable)
```

Sets the loss_weight attribute of the instance to the given function.

**Arguments**:

- `fn` - The function to set as the loss_weight attribute. This function should take three arguments: raw_loss, t, and weight_type.

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMloss_weight"></a>

#### loss\_weight

```python
def loss_weight(raw_loss: Tensor,
                t: Tensor,
                weight_type: str,
                dt: Float = 0.001) -> Tensor
```

Calculates the weight for the loss based on the given weight type.

This function computes the loss weight according to the specified `weight_type`.
The available weight types are:
- "ones": uniform weight of 1.0
- "data_to_noise": derived from Equation (9) of https://arxiv.org/pdf/2202.00512
- "variational_objective": based on the variational objective, see https://arxiv.org/pdf/2202.00512

**Arguments**:

- `raw_loss` _Tensor_ - The raw loss calculated from the model prediction and target.
- `t` _Tensor_ - The time step.
- `weight_type` _str_ - The type of weight to use. Can be "ones", "data_to_noise", or "variational_objective".
- `dt` _Float, optional_ - The time step increment. Defaults to 0.001.


**Returns**:

- `Tensor` - The weight for the loss.


**Raises**:

- `ValueError` - If the weight type is not recognized.

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMloss"></a>

#### loss

```python
def loss(model_pred: Tensor,
         target: Tensor,
         t: Tensor,
         dt: Optional[Float] = 0.001,
         mask: Optional[Tensor] = None,
         weight_type: str = "ones")
```

Calculates the loss given the model prediction, target, and time.

**Arguments**:

- `model_pred` _Tensor_ - The predicted output from the model.
- `target` _Tensor_ - The target output for the model prediction.
- `t` _Tensor_ - The time at which the loss is calculated.
- `dt` _Optional[Float], optional_ - The time step increment. Defaults to 0.001.
- `mask` _Optional[Tensor], optional_ - The mask for the data point. Defaults to None.
- `weight_type` _str, optional_ - The type of weight to use for the loss. Can be "ones", "data_to_noise", or "variational_objective". Defaults to "ones".


**Returns**:

- `Tensor` - The calculated loss batch tensor.

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMstep_hybrid_sde"></a>

#### step\_hybrid\_sde

```python
def step_hybrid_sde(model_out: Tensor,
                    t: Tensor,
                    xt: Tensor,
                    dt: Tensor,
                    mask: Optional[Tensor] = None,
                    center: Bool = False,
                    temperature: Float = 1.0,
                    equilibrium_rate: Float = 0.0) -> Tensor
```

Do one step integration of Hybrid Langevin-Reverse Time SDE.

See section B.3 page 37 https://www.biorxiv.org/content/10.1101/2022.12.01.518682v1.full.pdf.
and https://github.com/generatebio/chroma/blob/929407c605013613941803c6113adefdccaad679/chroma/layers/structure/diffusion.py#L730

**Arguments**:

- `model_out` _Tensor_ - The output of the model.
- `xt` _Tensor_ - The current data point.
- `t` _Tensor_ - The current time step.
- `dt` _Tensor_ - The time step increment.
- `mask` _Optional[Tensor], optional_ - An optional mask to apply to the data. Defaults to None.
- `center` _bool, optional_ - Whether to center the data. Defaults to False.
- `temperature` _Float, optional_ - The temperature parameter for low temperature sampling. Defaults to 1.0.
- `equilibrium_rate` _Float, optional_ - The rate of Langevin equilibration.  Scales the amount of Langevin dynamics per unit time. Best values are in the range [1.0, 5.0]. Defaults to 0.0.


**Notes**:

  For all step functions that use the SDE formulation its important to note that we are moving backwards in time which corresponds to an apparent sign change.
  A clear example can be seen in slide 29 https://ernestryu.com/courses/FM/diffusion1.pdf.

<a id="mocointerpolantscontinuous_timecontinuousvdmVDMstep_ode"></a>

#### step\_ode

```python
def step_ode(model_out: Tensor,
             t: Tensor,
             xt: Tensor,
             dt: Tensor,
             mask: Optional[Tensor] = None,
             center: Bool = False,
             temperature: Float = 1.0) -> Tensor
```

Do one step integration of ODE.

See section B page 36 https://www.biorxiv.org/content/10.1101/2022.12.01.518682v1.full.pdf.
and https://github.com/generatebio/chroma/blob/929407c605013613941803c6113adefdccaad679/chroma/layers/structure/diffusion.py#L730

**Arguments**:

- `model_out` _Tensor_ - The output of the model.
- `xt` _Tensor_ - The current data point.
- `t` _Tensor_ - The current time step.
- `dt` _Tensor_ - The time step increment.
- `mask` _Optional[Tensor], optional_ - An optional mask to apply to the data. Defaults to None.
- `center` _bool, optional_ - Whether to center the data. Defaults to False.
- `temperature` _Float, optional_ - The temperature parameter for low temperature sampling. Defaults to 1.0.

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matching"></a>

# bionemo.moco.interpolants.continuous\_time.continuous.continuous\_flow\_matching

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcher"></a>

## ContinuousFlowMatcher Objects

```python
class ContinuousFlowMatcher(Interpolant)
```

A Continuous Flow Matching interpolant.

-------

**Examples**:

```python
>>> import torch
>>> from bionemo.bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
>>> from bionemo.bionemo.moco.distributions.time.uniform import UniformTimeDistribution
>>> from bionemo.bionemo.moco.interpolants.continuous_time.continuous.continuous_flow_matching import ContinuousFlowMatcher
>>> from bionemo.bionemo.moco.schedules.inference_time_schedules import LinearInferenceSchedule

flow_matcher = ContinuousFlowMatcher(
    time_distribution = UniformTimeDistribution(...),
    prior_distribution = GaussianPrior(...),
    )
model = Model(...)

# Training
for epoch in range(1000):
    data = data_loader.get(...)
    time = flow_matcher.sample_time(batch_size)
    noise = flow_matcher.sample_prior(data.shape)
    data, time, noise = flow_matcher.apply_ot(noise, data) # Optional, only for OT
    xt = flow_matcher.interpolate(data, time, noise)
    flow = flow_matcher.calculate_target(data, noise)

    u_pred = model(xt, time)
    loss = flow_matcher.loss(u_pred, flow)
    loss.backward()

# Generation
x_pred = flow_matcher.sample_prior(data.shape)
inference_sched = LinearInferenceSchedule(...)
for t in inference_sched.generate_schedule():
    time = inference_sched.pad_time(x_pred.shape[0], t)
    u_hat = model(x_pred, time)
    x_pred = flow_matcher.step(u_hat, x_pred, time)
return x_pred

```

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcher__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_distribution: TimeDistribution,
             prior_distribution: PriorDistribution,
             prediction_type: Union[PredictionType, str] = PredictionType.DATA,
             sigma: Float = 0,
             ot_type: Optional[Union[OptimalTransportType, str]] = None,
             ot_num_threads: int = 1,
             data_scale: Float = 1.0,
             device: Union[str, torch.device] = "cpu",
             rng_generator: Optional[torch.Generator] = None,
             eps: Float = 1e-5)
```

Initializes the Continuous Flow Matching interpolant.

**Arguments**:

- `time_distribution` _TimeDistribution_ - The distribution of time steps, used to sample time points for the diffusion process.
- `prior_distribution` _PriorDistribution_ - The prior distribution of the variable, used as the starting point for the diffusion process.
- `prediction_type` _PredictionType, optional_ - The type of prediction, either "flow" or another type. Defaults to PredictionType.DATA.
- `sigma` _Float, optional_ - The standard deviation of the Gaussian noise added to the interpolated data. Defaults to 0.
- `ot_type` _Optional[Union[OptimalTransportType, str]], optional_ - The type of optimal transport, if applicable. Defaults to None.
- `ot_num_threads` - Number of threads to use for OT solver. If "max", uses the maximum number of threads. Default is 1.
- `data_scale` _Float, optional_ - The scale factor for the data. Defaults to 1.0.
- `device` _Union[str, torch.device], optional_ - The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
- `eps` - Small float to prevent divide by zero

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherapply_ot"></a>

#### apply\_ot

```python
def apply_ot(x0: Tensor,
             x1: Tensor,
             mask: Optional[Tensor] = None,
             **kwargs) -> tuple
```

Sample and apply the optimal transport plan between batched (and masked) x0 and x1.

**Arguments**:

- `x0` _Tensor_ - shape (bs, *dim), noise from source minibatch.
- `x1` _Tensor_ - shape (bs, *dim), data from source minibatch.
- `mask` _Optional[Tensor], optional_ - mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.
- `**kwargs` - Additional keyword arguments to be passed to self.ot_sampler.apply_ot or handled within this method.



**Returns**:

- `Tuple` - tuple of 2 tensors, represents the noise and data samples following OT plan pi.

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherundo_scale_data"></a>

#### undo\_scale\_data

```python
def undo_scale_data(data: Tensor) -> Tensor
```

Downscale the input data by the data scale factor.

**Arguments**:

- `data` _Tensor_ - The input data to downscale.


**Returns**:

  The downscaled data.

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherscale_data"></a>

#### scale\_data

```python
def scale_data(data: Tensor) -> Tensor
```

Upscale the input data by the data scale factor.

**Arguments**:

- `data` _Tensor_ - The input data to upscale.


**Returns**:

  The upscaled data.

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherinterpolate"></a>

#### interpolate

```python
def interpolate(data: Tensor, t: Tensor, noise: Tensor) -> Tensor
```

Get x_t with given time t from noise (x_0) and data (x_1).

Currently, we use the linear interpolation as defined in:
1. Rectified flow: https://arxiv.org/abs/2209.03003.
2. Conditional flow matching: https://arxiv.org/abs/2210.02747 (called conditional optimal transport).

**Arguments**:

- `noise` _Tensor_ - noise from prior(), shape (batchsize, nodes, features)
- `t` _Tensor_ - time, shape (batchsize)
- `data` _Tensor_ - target, shape (batchsize, nodes, features)

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatchercalculate_target"></a>

#### calculate\_target

```python
def calculate_target(data: Tensor,
                     noise: Tensor,
                     mask: Optional[Tensor] = None) -> Tensor
```

Get the target vector field at time t.

**Arguments**:

- `noise` _Tensor_ - noise from prior(), shape (batchsize, nodes, features)
- `data` _Tensor_ - target, shape (batchsize, nodes, features)
- `mask` _Optional[Tensor], optional_ - mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.


**Returns**:

- `Tensor` - The target vector field at time t.

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherprocess_vector_field_prediction"></a>

#### process\_vector\_field\_prediction

```python
def process_vector_field_prediction(model_output: Tensor,
                                    xt: Optional[Tensor] = None,
                                    t: Optional[Tensor] = None,
                                    mask: Optional[Tensor] = None)
```

Process the model output based on the prediction type to calculate vecotr field.

**Arguments**:

- `model_output` _Tensor_ - The output of the model.
- `xt` _Tensor_ - The input sample.
- `t` _Tensor_ - The time step.
- `mask` _Optional[Tensor], optional_ - An optional mask to apply to the model output. Defaults to None.


**Returns**:

  The vector field prediction based on the prediction type.


**Raises**:

- `ValueError` - If the prediction type is not "flow" or "data".

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherprocess_data_prediction"></a>

#### process\_data\_prediction

```python
def process_data_prediction(model_output: Tensor,
                            xt: Optional[Tensor] = None,
                            t: Optional[Tensor] = None,
                            mask: Optional[Tensor] = None)
```

Process the model output based on the prediction type to generate clean data.

**Arguments**:

- `model_output` _Tensor_ - The output of the model.
- `xt` _Tensor_ - The input sample.
- `t` _Tensor_ - The time step.
- `mask` _Optional[Tensor], optional_ - An optional mask to apply to the model output. Defaults to None.


**Returns**:

  The data prediction based on the prediction type.


**Raises**:

- `ValueError` - If the prediction type is not "flow".

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherstep"></a>

#### step

```python
def step(model_out: Tensor,
         xt: Tensor,
         dt: Tensor,
         t: Optional[Tensor] = None,
         mask: Optional[Tensor] = None,
         center: Bool = False)
```

Perform a single ODE step integration using Euler method.

**Arguments**:

- `model_out` _Tensor_ - The output of the model at the current time step.
- `xt` _Tensor_ - The current intermediate state.
- `dt` _Tensor_ - The time step size.
- `t` _Tensor, optional_ - The current time. Defaults to None.
- `mask` _Optional[Tensor], optional_ - A mask to apply to the model output. Defaults to None.
- `center` _Bool, optional_ - Whether to center the output. Defaults to False.


**Returns**:

- `x_next` _Tensor_ - The updated state of the system after the single step, x_(t+dt).


**Notes**:

  - If a mask is provided, it is applied element-wise to the model output before scaling.
  - The `clean` method is called on the updated state before it is returned.

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherstep_score_stochastic"></a>

#### step\_score\_stochastic

```python
def step_score_stochastic(model_out: Tensor,
                          xt: Tensor,
                          dt: Tensor,
                          t: Tensor,
                          mask: Optional[Tensor] = None,
                          gt_mode: str = "tan",
                          gt_p: Float = 1.0,
                          gt_clamp: Optional[Float] = None,
                          score_temperature: Float = 1.0,
                          noise_temperature: Float = 1.0,
                          t_lim_ode: Float = 0.99,
                          center: Bool = False)
```

Perform a single ODE step integration using Euler method.

d x_t = [v(x_t, t) + g(t) * s(x_t, t) * sc_score_scale] dt + \sqrt{2 * g(t) * temperature} dw_t.

At the moment we do not scale the vector field v but this can be added with sc_score_scale.

**Arguments**:

- `model_out` _Tensor_ - The output of the model at the current time step.
- `xt` _Tensor_ - The current intermediate state.
- `dt` _Tensor_ - The time step size.
- `t` _Tensor, optional_ - The current time. Defaults to None.
- `mask` _Optional[Tensor], optional_ - A mask to apply to the model output. Defaults to None.
- `gt_mode` _str, optional_ - The mode for the gt function. Defaults to "1/t".
- `gt_p` _Float, optional_ - The parameter for the gt function. Defaults to 1.0.
- `gt_clamp` - (Float, optional): Upper limit of gt term. Defaults to None.
- `score_temperature` _Float, optional_ - The temperature for the score part of the step. Defaults to 1.0.
- `noise_temperature` _Float, optional_ - The temperature for the stochastic part of the step. Defaults to 1.0.
- `t_lim_ode` _Float, optional_ - The time limit for the ODE step. Defaults to 0.99.
- `center` _Bool, optional_ - Whether to center the output. Defaults to False.


**Returns**:

- `x_next` _Tensor_ - The updated state of the system after the single step, x_(t+dt).


**Notes**:

  - If a mask is provided, it is applied element-wise to the model output before scaling.
  - The `clean` method is called on the updated state before it is returned.

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherloss"></a>

#### loss

```python
def loss(model_pred: Tensor,
         target: Tensor,
         t: Optional[Tensor] = None,
         xt: Optional[Tensor] = None,
         mask: Optional[Tensor] = None,
         target_type: Union[PredictionType, str] = PredictionType.DATA)
```

Calculate the loss given the model prediction, data sample, time, and mask.

If target_type is FLOW loss = ||v_hat - (x1-x0)||**2
If target_type is DATA loss = ||x1_hat - x1||**2 * 1 / (1 - t)**2 as the target vector field = x1 - x0 = (1/(1-t)) * x1 - xt where xt = tx1 - (1-t)x0.
This functions supports any cominbation of prediction_type and target_type in {DATA, FLOW}.

**Arguments**:

- `model_pred` _Tensor_ - The predicted output from the model.
- `target` _Tensor_ - The target output for the model prediction.
- `t` _Optional[Tensor], optional_ - The time for the model prediction. Defaults to None.
- `xt` _Optional[Tensor], optional_ - The interpolated data. Defaults to None.
- `mask` _Optional[Tensor], optional_ - The mask for the data point. Defaults to None.
- `target_type` _PredictionType, optional_ - The type of the target output. Defaults to PredictionType.DATA.


**Returns**:

- `Tensor` - The calculated loss batch tensor.

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatchervf_to_score"></a>

#### vf\_to\_score

```python
def vf_to_score(x_t: Tensor, v: Tensor, t: Tensor) -> Tensor
```

From Geffner et al. Computes score of noisy density given the vector field learned by flow matching.

With our interpolation scheme these are related by

v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

or equivalently,

s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

with scale_ref = 1

**Arguments**:

- `x_t` - Noisy sample, shape [*, dim]
- `v` - Vector field, shape [*, dim]
- `t` - Interpolation time, shape [*] (must be < 1)


**Returns**:

  Score of intermediate density, shape [*, dim].

<a id="mocointerpolantscontinuous_timecontinuouscontinuous_flow_matchingContinuousFlowMatcherget_gt"></a>

#### get\_gt

```python
def get_gt(t: Tensor,
           mode: str = "tan",
           param: float = 1.0,
           clamp_val: Optional[float] = None,
           eps: float = 1e-2) -> Tensor
```

From Geffner et al. Computes gt for different modes.

**Arguments**:

- `t` - times where we'll evaluate, covers [0, 1), shape [nsteps]
- `mode` - "us" or "tan"
- `param` - parameterized transformation
- `clamp_val` - value to clamp gt, no clamping if None
- `eps` - small value leave as it is

<a id="mocointerpolantscontinuous_time"></a>

# bionemo.moco.interpolants.continuous\_time

<a id="mocointerpolants"></a>

# bionemo.moco.interpolants

<a id="mocointerpolantsbatch_augmentation"></a>

# bionemo.moco.interpolants.batch\_augmentation

<a id="mocointerpolantsbatch_augmentationBatchAugmentation"></a>

## BatchAugmentation Objects

```python
class BatchAugmentation()
```

Facilitates the creation of batch augmentation objects based on specified optimal transport types.

**Arguments**:

- `device` _str_ - The device to use for computations (e.g., 'cpu', 'cuda').
- `num_threads` _int_ - The number of threads to utilize.

<a id="mocointerpolantsbatch_augmentationBatchAugmentation__init__"></a>

#### \_\_init\_\_

```python
def __init__(device, num_threads)
```

Initializes a BatchAugmentation instance.

**Arguments**:

- `device` _str_ - Device for computation.
- `num_threads` _int_ - Number of threads to use.

<a id="mocointerpolantsbatch_augmentationBatchAugmentationcreate"></a>

#### create

```python
def create(method_type: OptimalTransportType)
```

Creates a batch augmentation object of the specified type.

**Arguments**:

- `method_type` _OptimalTransportType_ - The type of optimal transport method.


**Returns**:

  The augmentation object if the type is supported, otherwise **None**.

<a id="mocointerpolantsdiscrete_timediscreted3pm"></a>

# bionemo.moco.interpolants.discrete\_time.discrete.d3pm

<a id="mocointerpolantsdiscrete_timediscreted3pmD3PM"></a>

## D3PM Objects

```python
class D3PM(Interpolant)
```

A Discrete Denoising Diffusion Probabilistic Model (D3PM) interpolant.

<a id="mocointerpolantsdiscrete_timediscreted3pmD3PM__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_distribution: TimeDistribution,
             prior_distribution: DiscretePriorDistribution,
             noise_schedule: DiscreteNoiseSchedule,
             device: str = "cpu",
             last_time_idx: int = 0,
             rng_generator: Optional[torch.Generator] = None)
```

Initializes the D3PM interpolant.

**Arguments**:

- `time_distribution` _TimeDistribution_ - The distribution of time steps, used to sample time points for the diffusion process.
- `prior_distribution` _PriorDistribution_ - The prior distribution of the variable, used as the starting point for the diffusion process.
- `noise_schedule` _DiscreteNoiseSchedule_ - The schedule of noise, defining the amount of noise added at each time step.
- `device` _str, optional_ - The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
- `last_time_idx` _int, optional_ - The last time index to consider in the interpolation process. Defaults to 0.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocointerpolantsdiscrete_timediscreted3pmD3PMinterpolate"></a>

#### interpolate

```python
def interpolate(data: Tensor, t: Tensor)
```

Interpolate using discrete interpolation method.

This method implements Equation 2 from the D3PM paper (https://arxiv.org/pdf/2107.03006), which
calculates the interpolated discrete state `xt` at time `t` given the input data and noise
via q(xt|x0) = Cat(xt; p = x0*Qt_bar).

**Arguments**:

- `data` _Tensor_ - The input data to be interpolated.
- `t` _Tensor_ - The time step at which to interpolate.


**Returns**:

- `Tensor` - The interpolated discrete state `xt` at time `t`.

<a id="mocointerpolantsdiscrete_timediscreted3pmD3PMforward_process"></a>

#### forward\_process

```python
def forward_process(data: Tensor, t: Tensor) -> Tensor
```

Apply the forward process to the data at time t.

**Arguments**:

- `data` _Tensor_ - target discrete ids
- `t` _Tensor_ - time


**Returns**:

- `Tensor` - x(t) after applying the forward process

<a id="mocointerpolantsdiscrete_timediscreted3pmD3PMstep"></a>

#### step

```python
def step(model_out: Tensor,
         t: Tensor,
         xt: Tensor,
         mask: Optional[Tensor] = None,
         temperature: Float = 1.0,
         model_out_is_logits: bool = True)
```

Perform a single step in the discrete interpolant method, transitioning from the current discrete state `xt` at time `t` to the next state.

This step involves:

1. Computing the predicted q-posterior logits using the model output `model_out` and the current state `xt` at time `t`.
2. Sampling the next state from the predicted q-posterior distribution using the Gumbel-Softmax trick.

**Arguments**:

- `model_out` _Tensor_ - The output of the model at the current time step, which is used to compute the predicted q-posterior logits.
- `t` _Tensor_ - The current time step, which is used to index into the transition matrices and compute the predicted q-posterior logits.
- `xt` _Tensor_ - The current discrete state at time `t`, which is used to compute the predicted q-posterior logits and sample the next state.
- `mask` _Optional[Tensor], optional_ - An optional mask to apply to the next state, which can be used to mask out certain tokens or regions. Defaults to None.
- `temperature` _Float, optional_ - The temperature to use for the Gumbel-Softmax trick, which controls the randomness of the sampling process. Defaults to 1.0.
- `model_out_is_logits` _bool, optional_ - A flag indicating whether the model output is already in logits form. If True, the output is assumed to be logits; otherwise, it is converted to logits. Defaults to True.


**Returns**:

- `Tensor` - The next discrete state at time `t-1`.

<a id="mocointerpolantsdiscrete_timediscreted3pmD3PMloss"></a>

#### loss

```python
def loss(logits: Tensor,
         target: Tensor,
         xt: Tensor,
         time: Tensor,
         mask: Optional[Tensor] = None,
         vb_scale: Float = 0.0)
```

Calculate the cross-entropy loss between the model prediction and the target output.

The loss is calculated between the batch x node x class logits and the target batch x node. If a mask is provided, the loss is
calculated only for the non-masked elements. Additionally, if vb_scale is greater than 0, the variational lower bound loss is
calculated and added to the total loss.

**Arguments**:

- `logits` _Tensor_ - The predicted output from the model, with shape batch x node x class.
- `target` _Tensor_ - The target output for the model prediction, with shape batch x node.
- `xt` _Tensor_ - The current data point.
- `time` _Tensor_ - The time at which the loss is calculated.
- `mask` _Optional[Tensor], optional_ - The mask for the data point. Defaults to None.
- `vb_scale` _Float, optional_ - The scale factor for the variational lower bound loss. Defaults to 0.0.


**Returns**:

- `Tensor` - The calculated loss tensor. If aggregate is True, the loss and variational lower bound loss are aggregated and
  returned as a single tensor. Otherwise, the loss and variational lower bound loss are returned as separate tensors.

<a id="mocointerpolantsdiscrete_timediscrete"></a>

# bionemo.moco.interpolants.discrete\_time.discrete

<a id="mocointerpolantsdiscrete_timecontinuousddpm"></a>

# bionemo.moco.interpolants.discrete\_time.continuous.ddpm

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPM"></a>

## DDPM Objects

```python
class DDPM(Interpolant)
```

A Denoising Diffusion Probabilistic Model (DDPM) interpolant.

-------

**Examples**:

```python
>>> import torch
>>> from bionemo.bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
>>> from bionemo.bionemo.moco.distributions.time.uniform import UniformTimeDistribution
>>> from bionemo.bionemo.moco.interpolants.discrete_time.continuous.ddpm import DDPM
>>> from bionemo.bionemo.moco.schedules.noise.discrete_noise_schedules import DiscreteCosineNoiseSchedule
>>> from bionemo.bionemo.moco.schedules.inference_time_schedules import DiscreteLinearInferenceSchedule


ddpm = DDPM(
    time_distribution = UniformTimeDistribution(discrete_time = True,...),
    prior_distribution = GaussianPrior(...),
    noise_schedule = DiscreteCosineNoiseSchedule(...),
    )
model = Model(...)

# Training
for epoch in range(1000):
    data = data_loader.get(...)
    time = ddpm.sample_time(batch_size)
    noise = ddpm.sample_prior(data.shape)
    xt = ddpm.interpolate(data, noise, time)

    x_pred = model(xt, time)
    loss = ddpm.loss(x_pred, data, time)
    loss.backward()

# Generation
x_pred = ddpm.sample_prior(data.shape)
for t in DiscreteLinearTimeSchedule(...).generate_schedule():
    time = torch.full((batch_size,), t)
    x_hat = model(x_pred, time)
    x_pred = ddpm.step(x_hat, time, x_pred)
return x_pred

```

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPM__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_distribution: TimeDistribution,
             prior_distribution: PriorDistribution,
             noise_schedule: DiscreteNoiseSchedule,
             prediction_type: Union[PredictionType, str] = PredictionType.DATA,
             device: Union[str, torch.device] = "cpu",
             last_time_idx: int = 0,
             rng_generator: Optional[torch.Generator] = None)
```

Initializes the DDPM interpolant.

**Arguments**:

- `time_distribution` _TimeDistribution_ - The distribution of time steps, used to sample time points for the diffusion process.
- `prior_distribution` _PriorDistribution_ - The prior distribution of the variable, used as the starting point for the diffusion process.
- `noise_schedule` _DiscreteNoiseSchedule_ - The schedule of noise, defining the amount of noise added at each time step.
- `prediction_type` _PredictionType_ - The type of prediction, either "data" or another type. Defaults to "data".
- `device` _str_ - The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
- `last_time_idx` _int, optional_ - The last time index for discrete time. Set to 0 if discrete time is T-1, ..., 0 or 1 if T, ..., 1. Defaults to 0.
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMforward_data_schedule"></a>

#### forward\_data\_schedule

```python
@property
def forward_data_schedule() -> torch.Tensor
```

Returns the forward data schedule.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMforward_noise_schedule"></a>

#### forward\_noise\_schedule

```python
@property
def forward_noise_schedule() -> torch.Tensor
```

Returns the forward noise schedule.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMreverse_data_schedule"></a>

#### reverse\_data\_schedule

```python
@property
def reverse_data_schedule() -> torch.Tensor
```

Returns the reverse data schedule.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMreverse_noise_schedule"></a>

#### reverse\_noise\_schedule

```python
@property
def reverse_noise_schedule() -> torch.Tensor
```

Returns the reverse noise schedule.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMlog_var"></a>

#### log\_var

```python
@property
def log_var() -> torch.Tensor
```

Returns the log variance.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMalpha_bar"></a>

#### alpha\_bar

```python
@property
def alpha_bar() -> torch.Tensor
```

Returns the alpha bar values.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMalpha_bar_prev"></a>

#### alpha\_bar\_prev

```python
@property
def alpha_bar_prev() -> torch.Tensor
```

Returns the previous alpha bar values.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMinterpolate"></a>

#### interpolate

```python
def interpolate(data: Tensor, t: Tensor, noise: Tensor)
```

Get x(t) with given time t from noise and data.

**Arguments**:

- `data` _Tensor_ - target
- `t` _Tensor_ - time
- `noise` _Tensor_ - noise from prior()

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMforward_process"></a>

#### forward\_process

```python
def forward_process(data: Tensor, t: Tensor, noise: Optional[Tensor] = None)
```

Get x(t) with given time t from noise and data.

**Arguments**:

- `data` _Tensor_ - target
- `t` _Tensor_ - time
- `noise` _Tensor, optional_ - noise from prior(). Defaults to None.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMprocess_data_prediction"></a>

#### process\_data\_prediction

```python
def process_data_prediction(model_output: Tensor, sample: Tensor, t: Tensor)
```

Converts the model output to a data prediction based on the prediction type.

This conversion stems from the Progressive Distillation for Fast Sampling of Diffusion Models https://arxiv.org/pdf/2202.00512.
Given the model output and the sample, we convert the output to a data prediction based on the prediction type.
The conversion formulas are as follows:
- For "noise" prediction type: `pred_data = (sample - noise_scale * model_output) / data_scale`
- For "data" prediction type: `pred_data = model_output`
- For "v_prediction" prediction type: `pred_data = data_scale * sample - noise_scale * model_output`

**Arguments**:

- `model_output` _Tensor_ - The output of the model.
- `sample` _Tensor_ - The input sample.
- `t` _Tensor_ - The time step.


**Returns**:

  The data prediction based on the prediction type.


**Raises**:

- `ValueError` - If the prediction type is not one of "noise", "data", or "v_prediction".

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMprocess_noise_prediction"></a>

#### process\_noise\_prediction

```python
def process_noise_prediction(model_output, sample, t)
```

Do the same as process_data_prediction but take the model output and convert to nosie.

**Arguments**:

- `model_output` - The output of the model.
- `sample` - The input sample.
- `t` - The time step.


**Returns**:

  The input as noise if the prediction type is "noise".


**Raises**:

- `ValueError` - If the prediction type is not "noise".

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMcalculate_velocity"></a>

#### calculate\_velocity

```python
def calculate_velocity(data: Tensor, t: Tensor, noise: Tensor) -> Tensor
```

Calculate the velocity term given the data, time step, and noise.

**Arguments**:

- `data` _Tensor_ - The input data.
- `t` _Tensor_ - The current time step.
- `noise` _Tensor_ - The noise term.


**Returns**:

- `Tensor` - The calculated velocity term.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMstep"></a>

#### step

```python
@torch.no_grad()
def step(model_out: Tensor,
         t: Tensor,
         xt: Tensor,
         mask: Optional[Tensor] = None,
         center: Bool = False,
         temperature: Float = 1.0)
```

Do one step integration.

**Arguments**:

- `model_out` _Tensor_ - The output of the model.
- `t` _Tensor_ - The current time step.
- `xt` _Tensor_ - The current data point.
- `mask` _Optional[Tensor], optional_ - An optional mask to apply to the data. Defaults to None.
- `center` _bool, optional_ - Whether to center the data. Defaults to False.
- `temperature` _Float, optional_ - The temperature parameter for low temperature sampling. Defaults to 1.0.


**Notes**:

  The temperature parameter controls the level of randomness in the sampling process. A temperature of 1.0 corresponds to standard diffusion sampling, while lower temperatures (e.g. 0.5, 0.2) result in less random and more deterministic samples. This can be useful for tasks that require more control over the generation process.

  Note for discrete time we sample from [T-1, ..., 1, 0] for T steps so we sample t = 0 hence the mask.
  For continuous time we start from [1, 1 -dt, ..., dt] for T steps where s = t - 1 when t = 0 i.e dt is then 0

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMstep_noise"></a>

#### step\_noise

```python
def step_noise(model_out: Tensor,
               t: Tensor,
               xt: Tensor,
               mask: Optional[Tensor] = None,
               center: Bool = False,
               temperature: Float = 1.0)
```

Do one step integration.

**Arguments**:

- `model_out` _Tensor_ - The output of the model.
- `t` _Tensor_ - The current time step.
- `xt` _Tensor_ - The current data point.
- `mask` _Optional[Tensor], optional_ - An optional mask to apply to the data. Defaults to None.
- `center` _bool, optional_ - Whether to center the data. Defaults to False.
- `temperature` _Float, optional_ - The temperature parameter for low temperature sampling. Defaults to 1.0.


**Notes**:

  The temperature parameter controls the level of randomness in the sampling process. A temperature of 1.0 corresponds to standard diffusion sampling, while lower temperatures (e.g. 0.5, 0.2) result in less random and more deterministic samples. This can be useful for tasks that require more control over the generation process.

  Note for discrete time we sample from [T-1, ..., 1, 0] for T steps so we sample t = 0 hence the mask.
  For continuous time we start from [1, 1 -dt, ..., dt] for T steps where s = t - 1 when t = 0 i.e dt is then 0

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMscore"></a>

#### score

```python
def score(x_hat: Tensor, xt: Tensor, t: Tensor)
```

Converts the data prediction to the estimated score function.

**Arguments**:

- `x_hat` _Tensor_ - The predicted data point.
- `xt` _Tensor_ - The current data point.
- `t` _Tensor_ - The time step.


**Returns**:

  The estimated score function.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMstep_ddim"></a>

#### step\_ddim

```python
def step_ddim(model_out: Tensor,
              t: Tensor,
              xt: Tensor,
              mask: Optional[Tensor] = None,
              eta: Float = 0.0,
              center: Bool = False)
```

Do one step of DDIM sampling.

**Arguments**:

- `model_out` _Tensor_ - output of the model
- `t` _Tensor_ - current time step
- `xt` _Tensor_ - current data point
- `mask` _Optional[Tensor], optional_ - mask for the data point. Defaults to None.
- `eta` _Float, optional_ - DDIM sampling parameter. Defaults to 0.0.
- `center` _Bool, optional_ - whether to center the data point. Defaults to False.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMset_loss_weight_fn"></a>

#### set\_loss\_weight\_fn

```python
def set_loss_weight_fn(fn)
```

Sets the loss_weight attribute of the instance to the given function.

**Arguments**:

- `fn` - The function to set as the loss_weight attribute. This function should take three arguments: raw_loss, t, and weight_type.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMloss_weight"></a>

#### loss\_weight

```python
def loss_weight(raw_loss: Tensor, t: Optional[Tensor],
                weight_type: str) -> Tensor
```

Calculates the weight for the loss based on the given weight type.

These data_to_noise loss weights is derived in Equation (9) of https://arxiv.org/pdf/2202.00512.

**Arguments**:

- `raw_loss` _Tensor_ - The raw loss calculated from the model prediction and target.
- `t` _Tensor_ - The time step.
- `weight_type` _str_ - The type of weight to use. Can be "ones" or "data_to_noise" or "noise_to_data".


**Returns**:

- `Tensor` - The weight for the loss.


**Raises**:

- `ValueError` - If the weight type is not recognized.

<a id="mocointerpolantsdiscrete_timecontinuousddpmDDPMloss"></a>

#### loss

```python
def loss(model_pred: Tensor,
         target: Tensor,
         t: Optional[Tensor] = None,
         mask: Optional[Tensor] = None,
         weight_type: str = "ones")
```

Calculate the loss given the model prediction, data sample, and time.

**Arguments**:

- `model_pred` _Tensor_ - The predicted output from the model.
- `target` _Tensor_ - The target output for the model prediction.
- `t` _Tensor_ - The time at which the loss is calculated.
- `mask` _Optional[Tensor], optional_ - The mask for the data point. Defaults to None.
- `weight_type` _str, optional_ - The type of weight to use for the loss. Defaults to "ones".


**Returns**:

- `Tensor` - The calculated loss batch tensor.

<a id="mocointerpolantsdiscrete_timecontinuous"></a>

# bionemo.moco.interpolants.discrete\_time.continuous

<a id="mocointerpolantsdiscrete_time"></a>

# bionemo.moco.interpolants.discrete\_time

<a id="mocointerpolantsdiscrete_timeutils"></a>

# bionemo.moco.interpolants.discrete\_time.utils

<a id="mocointerpolantsdiscrete_timeutilssafe_index"></a>

#### safe\_index

```python
def safe_index(tensor: Tensor, index: Tensor, device: torch.device)
```

Safely indexes a tensor using a given index and returns the result on a specified device.

Note can implement forcing with  return tensor[index.to(tensor.device)].to(device) but has costly migration.

**Arguments**:

- `tensor` _Tensor_ - The tensor to be indexed.
- `index` _Tensor_ - The index to use for indexing the tensor.
- `device` _torch.device_ - The device on which the result should be returned.


**Returns**:

- `Tensor` - The indexed tensor on the specified device.


**Raises**:

- `ValueError` - If tensor, index, and device are not all on the same device.

<a id="mocointerpolantsbase_interpolant"></a>

# bionemo.moco.interpolants.base\_interpolant

<a id="mocointerpolantsbase_interpolantstring_to_enum"></a>

#### string\_to\_enum

```python
def string_to_enum(value: Union[str, AnyEnum],
                   enum_type: Type[AnyEnum]) -> AnyEnum
```

Converts a string to an enum value of the specified type. If the input is already an enum instance, it is returned as-is.

**Arguments**:

- `value` _Union[str, E]_ - The string to convert or an existing enum instance.
- `enum_type` _Type[E]_ - The enum type to convert to.


**Returns**:

- `E` - The corresponding enum value.


**Raises**:

- `ValueError` - If the string does not correspond to any enum member.

<a id="mocointerpolantsbase_interpolantpad_like"></a>

#### pad\_like

```python
def pad_like(source: Tensor, target: Tensor) -> Tensor
```

Pads the dimensions of the source tensor to match the dimensions of the target tensor.

**Arguments**:

- `source` _Tensor_ - The tensor to be padded.
- `target` _Tensor_ - The tensor that the source tensor should match in dimensions.


**Returns**:

- `Tensor` - The padded source tensor.


**Raises**:

- `ValueError` - If the source tensor has more dimensions than the target tensor.


**Example**:

  >>> source = torch.tensor([1, 2, 3])  # shape: (3,)
  >>> target = torch.tensor([[1, 2], [4, 5], [7, 8]])  # shape: (3, 2)
  >>> padded_source = pad_like(source, target)  # shape: (3, 1)

<a id="mocointerpolantsbase_interpolantPredictionType"></a>

## PredictionType Objects

```python
class PredictionType(Enum)
```

An enumeration representing the type of prediction a Denoising Diffusion Probabilistic Model (DDPM) can be used for.

DDPMs are versatile models that can be utilized for various prediction tasks, including:

- **Data**: Predicting the original data distribution from a noisy input.
- **Noise**: Predicting the noise that was added to the original data to obtain the input.
- **Velocity**: Predicting the velocity or rate of change of the data, particularly useful for modeling temporal dynamics.

These prediction types can be used to train neural networks for specific tasks, such as denoising, image synthesis, or time-series forecasting.

<a id="mocointerpolantsbase_interpolantInterpolant"></a>

## Interpolant Objects

```python
class Interpolant(ABC)
```

An abstract base class representing an Interpolant.

This class serves as a foundation for creating interpolants that can be used
in various applications, providing a basic structure and interface for
interpolation-related operations.

<a id="mocointerpolantsbase_interpolantInterpolant__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_distribution: TimeDistribution,
             prior_distribution: PriorDistribution,
             device: Union[str, torch.device] = "cpu",
             rng_generator: Optional[torch.Generator] = None)
```

Initializes the Interpolant class.

**Arguments**:

- `time_distribution` _TimeDistribution_ - The distribution of time steps.
- `prior_distribution` _PriorDistribution_ - The prior distribution of the variable.
- `device` _Union[str, torch.device], optional_ - The device on which to operate. Defaults to "cpu".
- `rng_generator` - An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

<a id="mocointerpolantsbase_interpolantInterpolantinterpolate"></a>

#### interpolate

```python
@abstractmethod
def interpolate(*args, **kwargs) -> Tensor
```

Get x(t) with given time t from noise and data.

Interpolate between x0 and x1 at the given time t.

<a id="mocointerpolantsbase_interpolantInterpolantstep"></a>

#### step

```python
@abstractmethod
def step(*args, **kwargs) -> Tensor
```

Do one step integration.

<a id="mocointerpolantsbase_interpolantInterpolantgeneral_step"></a>

#### general\_step

```python
def general_step(method_name: str, kwargs: dict)
```

Calls a step method of the class by its name, passing the provided keyword arguments.

**Arguments**:

- `method_name` _str_ - The name of the step method to call.
- `kwargs` _dict_ - Keyword arguments to pass to the step method.


**Returns**:

  The result of the step method call.


**Raises**:

- `ValueError` - If the provided method name does not start with 'step'.
- `Exception` - If the step method call fails. The error message includes a list of available step methods.


**Notes**:

  This method allows for dynamic invocation of step methods, providing flexibility in the class's usage.

<a id="mocointerpolantsbase_interpolantInterpolantsample_prior"></a>

#### sample\_prior

```python
def sample_prior(*args, **kwargs) -> Tensor
```

Sample from prior distribution.

This method generates a sample from the prior distribution specified by the
`prior_distribution` attribute.

**Returns**:

- `Tensor` - The generated sample from the prior distribution.

<a id="mocointerpolantsbase_interpolantInterpolantsample_time"></a>

#### sample\_time

```python
def sample_time(*args, **kwargs) -> Tensor
```

Sample from time distribution.

<a id="mocointerpolantsbase_interpolantInterpolantto_device"></a>

#### to\_device

```python
def to_device(device: str)
```

Moves all internal tensors to the specified device and updates the `self.device` attribute.

**Arguments**:

- `device` _str_ - The device to move the tensors to (e.g. "cpu", "cuda:0").


**Notes**:

  This method is used to transfer the internal state of the DDPM interpolant to a different device.
  It updates the `self.device` attribute to reflect the new device and moves all internal tensors to the specified device.

<a id="mocointerpolantsbase_interpolantInterpolantclean_mask_center"></a>

#### clean\_mask\_center

```python
def clean_mask_center(data: Tensor,
                      mask: Optional[Tensor] = None,
                      center: Bool = False) -> Tensor
```

Returns a clean tensor that has been masked and/or centered based on the function arguments.

**Arguments**:

- `data` - The input data with shape (..., nodes, features).
- `mask` - An optional mask to apply to the data with shape (..., nodes). If provided, it is used to calculate the CoM. Defaults to None.
- `center` - A boolean indicating whether to center the data around the calculated CoM. Defaults to False.


**Returns**:

  The data with shape (..., nodes, features) either centered around the CoM if `center` is True or unchanged if `center` is False.
