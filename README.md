## MaxCorr: A Package for Maximal Correlation Indicators

*MaxCorr* is a python package for the estimation of Maximal (Non-Linear) Correlations in sets of bivariate and multivariate data.

The package is available in PyPI and can be installed using the following command:
```
pip install maxcorr
```

Depending on the chosen algorithm and/or backend library to perform the numerical computation, additional dependencies might be required.
Specifically, such dependencies can be downloaded passing a parameter between square brackets during the installation, i.e.:
* `maxcorr[torch]`, includes also a version of PyTorch (required for `DensityIndicator` or any indicator using `torch` as backend);
* `maxcorr[tensorflow]`, includes also a version of Tensorflow (required for any indicator using `tensorflow` as backend);
* `maxcorr[lattice]`, includes also a version of Tensorflow and Tensorflow Lattice (required for `LatticeIndicator`).

A full installation including all the optional dependencies can be downloaded using:
```
pip install 'maxcorr[full]'
```

### **Available Semantics**

*MaxCorr* currently includes **three semantics** for non-linear correlations, namely:

[**HGR**]
The Hirschfield-Gebelin-Rényi coefficient ([Rényi, 1959](https://static.Rényi.hu/Rényi_cikkek/1959_on_measures_of_dependence.pdf)) is a non-linear correlation coefficient based on Pearson's correlation.
It is defined as the maximal correlation that can be achieved by transforming two random variables ($A, B$) into non-linear domains through two copula transformations, i.e.:

$$\text{HGR}(A, B) = \sup_{f \in \mathcal{F}, g \in \mathcal{F}} \frac{\text{cov}(f_A, g_B)}{\sigma(f_A) \cdot \sigma(f_B)}$$

where $f_A = f(A)$ and $g_B = g(B)$ are the two copula transformations belong to the Hilbert $\mathcal{F}$ space of all possible mapping functions.
HGR is proved to be uncomputable, yet many different estimation algorithms have been proposed in the literature, some of which are implemented in this package.

[**GeDI**]
The Generalized Disparate Impact ([Giuliani et al., 2023](https://proceedings.mlr.press/v202/giuliani23a/giuliani23a.pdf)) is a non-linear correlation coefficient which extends the concept of Disparate Impact ([Aghaei et al., 2019](https://dl.acm.org/doi/pdf/10.1609/aaai.v33i01.33011418)) defined in the field of algorithmic fairness and whose value is computed as the ratio between the covariance of the two variables $A$ and $B$ and the variance of $A$ – also known as $\beta$ coefficient in statistics and finance.
Although the original definition of the indicator from _Giuliani et al._ is specific to their proposed estimation approach, here we use a slight variation in order to link it to HGR and make it computable by all our implemented algorithms.
Formally, we define GeDI as:

$$\text{GeDI}(A, B) = \sup_{f \in \mathcal{F}, g \in \mathcal{F}} \frac{\text{cov}(f_A, g_B)}{\sigma(f_A)^2} \quad \text{s.t.} \quad \sigma(f_A) = \sigma(A), \quad \sigma(g_B) = \sigma(B)$$

namely we bound the two copula transformation to maintain the same standard deviation of the original random variables in order to avoid the explosion of the indicator by means of a simple rescaling of the mapping functions.
Since HGR is based on Pearson's correlation, hence it is scale invariant, this aspect was not problematic in its definition; however, this also allows us to redefine HGR by imposing the same constraints used in GeDI without loss of generality, i.e.:

$$\text{HGR}(A, B) = \sup_{f \in \mathcal{F}, g \in \mathcal{F}} \frac{\text{cov}(f_A, g_B)}{\sigma(f_A) \cdot \sigma(g_B)} \quad \text{s.t.} \quad \sigma(f_A) = \sigma(A), \quad \sigma(g_B) = \sigma(B)$$

making GeDI equivalent to HGR up to a scaling factor which depends on the standard deviation of the original random variables, i.e.:

$$\text{GeDI}(A, B) = \text{HGR}(A, B) \cdot \frac{\sigma(B)}{\sigma(A)}$$

[**NLC**]
The Non-Linear Covariance is a non-linear extension of the covariance measure that comes naturally after the definition of the first two semantics.
By leveraging the same constraints used in GeDI, we define NLC as:

$$\text{NLC}(A, B) = \sup_{f \in \mathcal{F}, g \in \mathcal{F}} \text{cov}(f_A, g_B) \quad \text{s.t.} \quad \sigma(f_A) = \sigma(A), \quad \sigma(g_B) = \sigma(B)$$

hence, adopting the same strategy used before, we have that NLC is also equivalent to HGR up to a scaling factor, i.e.:

$$\text{NLC}(A, B) = \text{HGR}(A, B) \cdot \sigma(A) \cdot \sigma(B)$$

### **Available Algorithms**

*MaxCorr* currently implements **six algorithms** to estimate the non-linear correlations:

[**Double Kernel**]
This algorithm is inspired by the one proposed by [Giuliani et al. (2023)](https://proceedings.mlr.press/v202/giuliani23a/giuliani23a.pdf) for the computation of the Generalized Disparate Impact, and extended to account for HGR computation as well.
Given two vectors $a$ and $b$, the core idea is to build two kernel matrices $F_a = [f_1(a), ..., f_h(a)]$ and $G_b = [g_1(b), ... g_k(b)]$ based on a set of user-defined functions.
The copula transformations $f$ and $g$ are then represented as the matrix product $F_a \cdot \alpha$ and $G_b \cdot \beta$, respectively, where $\alpha$ and $\beta$ are two vectors of mixing coefficients.
Eventually, the algorithm then uses global optimization techniques to find the optimal coefficient vectors resulting in the maximal correlation.
When using `torch` or `tensorflow` as backends, the algorithm also return gradient information along with the solution; moreover, if no mapping functions are specified, the indicator uses polynomial kernel expansions $F_a = [a, a^2, ..., a^h]$ and $G_b = [b, b^2, ..., b^k]$. 

[**Single Kernel**]
This is a variant of the previous algorithm which allows to account for functional dependencies only, although in either directions.
Formally, given a set of functions $f_1, ..., f_d$, tests the correlation between $[f_1(a), ..., f_d(a)] \cdot \alpha$ and $b$, and between $a$ and $[f_1(b), ..., f_d(b)] \cdot \beta$, eventually taking the maximal value.
This allows to find the optimal results by solving an unconstrained least-square problem, which is both computationally less demanding than global optimization methods and also allows to obtain a proper gradient rather than a sub-gradient when using `torch` or `tensorflow` as backends.
Again, if no mapping functions are specified, the indicator uses polynomial kernel expansions.

[**Neural**]
This approach was proposed by [Grari et al. (2020)](https://www.ijcai.org/proceedings/2020/0313.pdf) and models the two copula transformation $f$ and $g$ as two neural networks.
The algorithm works natively with both `torch` and `tensorflow` backends, hence it can return gradient information in both cases.
However, in order to function, it is necessary to install at least one of the two libraries required to perform the neural training, even in case of `numpy` backend.

[**Lattice**]
This is a custom variant of Grari et al.'s approach which uses Lattice Models to approximate the two copula transformation.
In order to use this algorithm, the additional dependency `[lattice]` must be added when installing the package; moreover, since the computational method relies on Tensorflow Lattice, gradient information is only returned when using `tensorflow` as backend.

[**Density**]
This approach was proposed by [Mary et al. (2019)](https://proceedings.mlr.press/v97/mary19a/mary19a.pdf) and computes the maximal correlation using a theoretical upper-bound of HGR known as Witsenhausen's characterization ([Witsenhausen, 1975](https://www.jstor.org/stable/pdf/2100465.pdf)).
Since the algorithm needs to perform Kernel-Density Estimation using `torch` primitives, PyTorch needs to be installed in the machine and gradient information is only returned when using `torch` as backend.

[**Randomized**]
This approach was proposed by [Lopez-Paz et al. (2013)](https://papers.nips.cc/paper_files/paper/2013/file/aab3238922bcc25a6f606eb525ffdc56-Paper.pdf) and computes the maximal correlation by mapping the input vectors $a$ and $b$ with a set of randomly-calibrated sinusoidal functions.
Eventually, the pair of functions returning the highest correlation is selected using Canonical Correlation Analysis (CCA); nonetheless, this algorithm cannot provide any gradient information.

### **Usage**

Indicators can be built using the main function `maxcorr.indicator`, specifying these three options.
Eventually, the correlation can be computed using the method `compute(a, b)` method, passing the inputs vectors/matrices as parameters.

```python
import numpy as np
from maxcorr import indicator

a = np.random.normal(loc=3.0, scale=2.0, size=100)
b = np.random.normal(loc=2.0, scale=5.0, size=100)
ind = indicator(semantics='hgr', algorithm='dk', backend='numpy')
ind.compute(a, b)
```

Moreover, algorithms might have specific parameters that can be passed as keyword arguments to the `indicator` function, or by calling the constructor method of each respective class which explicitly asks for specific indicator parameters.
```python
from maxcorr import indicator
from maxcorr.indicators import DoubleKernelIndicator

dk1 = indicator(algorithm='dk', kernel_a=5, kernel_b=1)
dk2 = DoubleKernelIndicator(kernel_a=5, kernel_b=1)
```

For a more in-depth exposition, follow the tutorial file `tutorial.ipynb`.