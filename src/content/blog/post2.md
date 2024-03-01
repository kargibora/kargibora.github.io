---
title: "Influence Functions in XAI "
description: ""
pubDate: "Jan 15 2024"
heroImage: "https://media.istockphoto.com/id/1140819116/photo/math-and-geometry-concept.jpg?s=612x612&w=0&k=20&c=iegPAlpFbqAMI-GuaD7ly8BEYtizimLN3kxjcbWIX6w="
tags: ["XAI","Paper Review", "Influence Functions"]
layout: '../../layouts/MarkdownLayout.astro'
badge: ""
timeToRead: "15"
---

# 1. Introduction
In the ever-evolving field of deep learning, the need for understanding the model decision becomes more important each day. To fully incorprate deep learning systems into a critical decision systems such as self-driving cars and medical AI supervisors, tracing back the algorithm into its' data point gives us a lot of understanding about the decision system itself. However deep learning systems are *often* blackboxes. 

One simple question arises from these concerns:

<div style="text-align:center;">
    "What is the effect of a single data point on the decision?"
</div>

One can find how *influential* a data point $x$is on the decision of the model $f_\theta(x^*)$ where $x^*$is a point not seen in the dataset by asking:

<div style="text-align:center;">
    "What would happen if the training point was not in the dataset; or was perturbed a little bit?"
</div>

A simple way to calculate this effect is to retrain your model *without the data point* $x$: however this is a very costly approach as current deep learning models usually takes a long time to train.

*Influence Functions* approximates the effect of how *influential* a point is with other points. In this blog post, we will formally defining the influential functions, their mathematical formulations, their advantages and their problems. 

# 2. Influence Functions
Koh et Al. [^1] use *influence functions*, which is a powerful tool in statistics, to overcome this problem. Instead of *retraining approach*, they compute how *influential* a point is on the decision with the *influential functions*. Consider a function, which takes input from the input space $\mathcal X$, to an output space $\mathcal Y$ given the training points $(z_i,...,z_n)$ where $z_i = (x_i,y_i)\in \mathcal X \times \mathcal Y$. Define an arbitrary lose function $L$ defined with the model parameters $\theta \in \Theta$ where
$$
L(z,\theta) = \frac{1}{n}\sum_{i=1}^n L(z_i,\theta) \tag{2.1}
$$

The paper also assumes about the *twice-differentiability* and *convexity* of the $\theta$. This will be important later on. Formally defining the influence of a point $x$, we have to define the change in model parameters we get after removing a point as:
$$
\hat \theta_{- z} - \hat \theta \quad \textnormal{where} \quad \hat \theta_{-z} = \argmin_{\theta \in \Theta}\sum_{z_i \neq z}L(z_i,\theta)  \tag{2.2}
$$
where $z = (x,y)$is the point we are removing, and $\hat \theta$ are the model parameters that minimizes the empirical risk $R(\theta) = \frac{1}{n}\sum_{i=1}^n L(z_i,\theta)$.

Instead of retraining to get $\hat \theta_{-z}$, what we if we approximate the result by checking the sensivity of a point $z$ to a weight in the loss? To be more formally, given $\epsilon$, define:
$$
\hat \theta_{\epsilon, z} = \argmin_{\theta \in \Theta} \frac{1}{n} \sum_{i=1}^{n}L(z_i, \theta) + \epsilon L(z,\theta)  \tag{2.3}
$$
If $\epsilon = \frac{-1}{n}$, this would be equalivent to removing the point $z$ from the training set. Notice that this is basically the model parameters after training the model with added weight to the $z$. Influence of upweighting $z$ can be calculated as:
$$
\mathcal I_{up,params}(z) = \frac{d\hat \theta_{\epsilon,z}}{d\epsilon} \big |_{\epsilon = 0}
$$

To derive the closed form of this derivative, rewrite the equation (2.3) as:
$$
\hat \theta_{\epsilon,z} = \argmin_{\theta \in \Theta} R(\theta) + \epsilon L(z,\theta) \tag{2.4}
$$

The quantity we seek $\frac{d\hat \theta_{\epsilon,z}}{d\epsilon}$ can be computed as:
$$
\frac{d\hat \theta_{\epsilon,z}}{d\epsilon} = \frac{d\Delta_\epsilon}{d\epsilon}
$$
where $\Delta_\epsilon = \hat \theta_{\epsilon,z} - \hat \theta$ as $\frac{d\hat \theta}{d\epsilon}=0$. Since $\hat \theta_{\epsilon,z}$ is the minimizer of the (2.4), first-order optimality condition holds:
$$
\nabla R(\hat \theta_{\epsilon, z}) + \epsilon \nabla L(z,\hat \theta_{\epsilon,z}) = 0
$$

Since $\lim_{\epsilon \rightarrow 0} L(z, \hat \theta_{\epsilon,z}) = L(z, \hat \theta)$ and thus, $\lim_{\epsilon \rightarrow 0} \hat \theta_{\epsilon,z} =  \hat \theta$, we can perform a taylor expension on the right-hand side of the equation $(2.4)$ to get approximation
$$
0 \approx [\nabla R(\hat \theta) + \epsilon \nabla L(z,\hat \theta)] + [\nabla^2 R(\hat \theta) + \epsilon \nabla^2 L(z,\hat \theta)] \Delta_\epsilon \tag{2.5}
$$
where the $o(||\nabla_\epsilon||)$ error term is dropped. Get the $\Delta_\epsilon$ approximation by readjusting the equation $(2.5)$:
$$
\begin{align*}
\Delta_\epsilon \approx - [\nabla^2 R(\hat \theta) + \epsilon \nabla^2 L(z,\hat \theta)]^{-1}[\nabla R(\hat \theta) + \epsilon \nabla L(z,\hat \theta)] \tag{2.6}
\end{align*}
$$
Since $\epsilon \rightarrow 0$ and $\nabla R(\hat \theta) = 0$, as $\hat \nabla$ minimizes the risk function $R(\theta)$, we can write the equation $(2.6)$as  <a id="eq:taylor"></a>
$$
\Delta_\epsilon \approx -\nabla^2 R(\hat \theta)^{-1} \nabla L(z,\hat \theta)\epsilon \tag{2.7} 
$$
where the $\epsilon \nabla^2 L(z,\hat \theta)$ is dropped as we drop the $o(\epsilon)$ term (assumption of twice differantibility).

Rewriting the $\nabla^2 R(\hat \theta) = H_{\hat \theta}$ where
$$
H_{\hat \theta} = \frac{1}{n}\sum_{i=1}^n \nabla_{\theta}^2 L(z_i,\hat \theta) \tag{2.8}
$$
we compute the $\mathcal I_{up,params}(z)$ as
$$
\begin{align*}
\mathcal I_{up,params}(z) = \frac{d\hat \theta_{\epsilon,z}}{d\epsilon} = \frac{d\Delta_\epsilon}{d\epsilon} \\
= -H_{\hat\theta}^{-1}\nabla L(z,\hat \theta)
\end{align*} \tag{2.9}
$$

Now the effect of upweighting the point $z$ is derived, let us calculate the effect of upweighting on the *loss function* by using the chain rule:
$$
\begin{align*}
\mathcal I_{up,loss}(z,z_{test}) = \frac{dL(z_{test}, \hat \theta_{\epsilon,z})}{d\epsilon} \big |_{\epsilon=0} \\
= \nabla_\theta L(z_{test}, \hat \theta)^T \frac{d\hat \theta_{\epsilon,z}}{d\epsilon} \big |_{\epsilon=0} \\
= -\nabla_\theta L(z_{test}, \hat \theta)^T H_{\hat\theta}^{-1}\nabla L(z,\hat \theta)
\end{align*} \tag{2.10}
$$
where ${z_{test}}$ is the test point and $z$ is the point that is *upweighted*. Basically, $\mathcal I_{up,loss}(z,z_{test})$ approximates the *influence of the point* $z$ *on the* $z_{test}$.

However the problem of this approach is, the inverse of Hessian is quite expensive to compute. This bring us to the next section.

## 2.1 Efficiently calculating the influence

Inverting the Hessian of the empirical risk takes $O(np^2 + p^3)$operations where $n$ is the number of points and $p$is the dimension of the weights $\theta \in \mathbb R^p$. To efficiently compute the inverse of a Hessian, authors use Hessian-vector products (HVPs) to efficiently approximate:
$$
s_{test} = H_{\hat \theta}^{-1} \nabla_\theta L(z_{test}, \hat \theta)
$$

If $s_{test}$is precomputed, calculating influence of a point $z$ on $z_{test}$ would be fast as we would only need to compute $\nabla L(z_{test},\hat \theta)$:
$$
\mathcal I_{up,loss}(z,z_{test}) = -s_{test} \cdot \nabla_\theta L(z_{test},\hat \theta)
$$

Authors discuss two algorithms in order to efficiently compute Hessian approximations. First idea is to use *conjugate gradients* method, in which the assumption of Hessian being a convex is used to transform the problem into a optimization problem. The second step [^4] uses the well known fact about the inverse of a convex matrix with $||A|| = 1$ to calculate the inverse of the matrix
$$
A^{-1} = \sum_{i=0}^\infty (I - A)^i
$$
iteratively. Let 
$$
H_j^{-1} = \sum_{i=0}^j (I - H)^i
$$
be the first $j$terms in the Taylor expension of $H^{-1}$. We can rewrite this equation as:
$$
H_j^{-1} = I + (I - H)H_{j-1}^{-1}
$$
Notice how we can compute this iteratively : by simply storing the previous estimation of the Hessian. After sampling $t$ points from a uniform distribution, authors suggest that $H$can be approximated with a single point $z_{s_j}$ and inverse hessian can be estimated as
$$
\tilde H_j^{-1} = I + (I - \nabla_{\theta}^2 L(z_{s_j},\hat \theta))\tilde H_{j-1}^{-1}
$$
where $\tilde H_0^{-1}v = v$ is set. With a large $t$, $\tilde H_t$ stabilizes. Authors also suggest to use this procedure $r$ times and take the average of the results. This algorithm can be used to compute $\mathcal I_{up,loss}(z_i,z_{test})$ in $O(np + rtp)$ time. To understand more about this sections, it is suggested to study the original LiSSA algorithm for estimating Hessian inverse [^4].

### 3. Experiments & Results

### 4. Faster Influence Functions

# 5. Fragility of Influence Functions 

We have covered some basic aspects of the influence functions. In practice, however, the positive definiteness and convexity assumption fails due to non-convex loss functions and complex structures. Also in most of the cases, exact Hessian is not computed we require an approximation. This bring us to the next topic of discussing whether influence functions are fairly accurate for more deeper, complex networks. 

Basu et al. [^5] works on the fragility of influence functions shows that hyperparameters and network structure greatly affects the quality of approximating the *re-training of a model without a particular input* via influence functions.  

Taylor approximation around the optimal parameters, as shown in the equation [(2.7)](#eq:taylor)
, can be inaccurate if the parameter space is varies a lot within its neighbourhood. Particularly, this happens for non-convex loss functions. Authors investigate the effects of the weight decay regularization, network depth and height. 

## 5.1 Experiments
The study investigates the effectiveness of influence functions across datasets of varying complexity, starting with the Iris dataset and advancing through MNIST, CIFAR-10, to ImageNet, to assess their accuracy in deep learning models. It evaluates these influence estimates using both Pearson and Spearman correlation methods, focusing on the latter for its relevance in ranking influential examples by importance. This approach allows for a detailed analysis of how well influence functions can identify and rank influential training points in relation to a specific test point, providing insights into their scalability and utility in interpreting deep learning models.

## 5.1.1 Understanding IF when the exact Hessian can be computed

The computation of the Hessian matrix and its inverse is a computationally intensive task, particularly in the context of large neural networks. Due to this complexity, iterative algorithms are often employed to approximate the Hessian and its inverse, providing a balance between computational feasibility and the accuracy of influence function (IF) estimates. In the initial experiments conducted on the Iris dataset, which features a small feed-forward neural network, the authors took advantage of the dataset's manageability to compute the exact Hessian. This approach is advantageous as it allows for a precise comparison between the influence functions' estimates and the true influence exerted by training points on the model's predictions. The exact computation of the Hessian in this context serves as a valuable baseline, offering clear insights into the accuracy and reliability of influence functions in simpler neural network settings before extending the analysis to more complex models and larger datasets where approximations are necessary.

<figure id="fig2">
    <img src="/src/assets/blog2/figure-2.png" width="720" height="720" alt="Figure-2">
    <figcaption>Figure 2: Iris dataset experimental results; (a) Spearman correlation of inﬂuence estimates with the ground-truth estimates computed with stochastic estimation vs. exact inverse-Hessian vector product. (b) Top eigenvalue of the Hessian vs. the network depth. (c) Spearman correlation between the norm of parameter changes computed with inﬂuence function vs. re-training. Figure taken from the original paper. </figcaption>
</figure>


#### 5.1.1.1 Effect of the Weight Decay
Weight decay is a common regularization technique that pushes the model towards a simpler hypothesis space. With their extensive experiments, authors show that *weight decay has a large effect on the high quality influence estimates*. To be more precise, they found that for the small feed-forward neural networks that is trained with weight-decay greatly increases the Spearman correlation between the influence estimates and the ground-truth estimates.

#### 5.1.1.2 Effect of the Depth
The study highlights the significant impact of network depth on the accuracy of influence estimates, observing that deeper networks (beyond 8 layers) exhibit a notable decline in Spearman correlation with ground-truth parameter changes. This decline suggests that as network depth increases, the ability of influence functions to accurately approximate parameter changes diminishes. The research quantifies this by comparing the norm of true parameter changes (obtained through re-training) against approximate changes predicted by influence functions, especially focusing on the most influential examples. A consistent trend emerges where the approximation error widens with network depth, particularly exceeding 5 layers, alongside an observed increase in the loss function's curvature. This finding underscores the challenges deeper networks pose to the precision of influence-based interpretability methods.

#### 5.1.1.3 Effect of the Width
The research examines the impact of increasing the width of a feed-forward network, while maintaining a constant depth, on the quality of influence estimates. It finds that as the network becomes wider, from 8 to 50 units, there's a consistent decrease in the Spearman correlation, from 0.82 to 0.56. This indicates that over-parameterization through wider networks detrimentally affects the accuracy of influence estimates, suggesting a strong relationship between network width and the reliability of these estimates in capturing the influence of training points on the model's predictions.

#### 5.1.1.4 Effect of the Inverse-Hessian Vector Product
As used by the authors of the original IF paper, using stochastic approximation makes the obtaining inverse of Hessian feasible, however it brings some approximation to the table. Authors suggest that stochastic estimation has a lower Spearman correlation across different model heights.

### 5.1.2 Understanding IF for Shallow CNN
In this case study, a Convolutional Neural Network (CNN) is utilized to analyze the small MNIST dataset, which consists of 10% of the full MNIST data, following a methodology similar to that of Koh & Liang (2017)[^1]. The focus is on evaluating the precision of influence estimates. This evaluation involves selecting test points that exhibit high test losses when the model is at its optimal parameters. For each chosen test point, the study identifies 100 training samples that have the highest influence scores and computes their ground-truth influence by re-training the model. Additionally, 100 training points are selected based on their influence scores being at the 30th percentile of the overall distribution, indicating they have low influence scores and less variability in those scores compared to the highly influential points. 

The study underscores the critical role of hyperparameter tuning, particularly weight decay, in enhancing the accuracy of influence estimates, similar to cases involving exact Hessian computations. It highlights that the selection of test points is a crucial factor influencing the quality of influence estimates. The research demonstrates variability in the effectiveness of influence estimates based on the chosen test points, with the correlation of influence estimates in a weight-decay trained network varying significantly, from 0.92 to 0.38, across different test points. This variability indicates a high sensitivity of the network to the choice of hyperparameters and the specifics of the training procedure, emphasizing the need for careful selection and tuning to obtain reliable influence estimate correlations.

### 5.1.3 Understanding IF for Deep Architectures
The accuracy of influence estimates is assessed using MNIST and CIFAR-10 datasets across various network architectures, including small CNNs, LeNet, ResNets, and VGGNets. For each architecture, two types of test points are selected for analysis: one with the highest loss and another at the median loss level among all test points. For both test points, the top 40 most influential training samples are identified, and the correlation between their influence estimates and the actual (ground-truth) influence is calculated. The ground-truth influence is determined by re-training the models from their optimal parameters for 6% of the original training steps, a method inspired by original IF paper [^1]. Additionally, all networks are trained with weight-decay regularization, applying a consistent weight-decay factor of 0.001 across different architectures to maintain uniformity in the evaluation process. Further details on this procedure are provided in the appendix of the study.

In the analysis of influence estimates across MNIST and CIFAR-10 datasets, it was observed that shallow networks on MNIST produce fairly accurate influence estimates, whereas the accuracy diminishes with increased network depth. For CIFAR-10, despite overall significant influence estimates, there's a slight drop in correlation for deeper networks such as ResNet-50. This relative improvement in CIFAR-10's influence estimates is attributed to CIFAR-10 trained architectures being less over-parameterized for a given depth compared to those trained on MNIST, aligning with findings that over-parameterization reduces influence estimate quality.

The study also emphasizes the considerable impact of test-point selection on the quality of influence estimates, with notable variations across different architectures. Small CNNs and LeNet architectures generally yield more accurate influence estimates, whereas the accuracy for ResNet-50 declines for both datasets. This variance suggests that the specific characteristics of loss landscapes at optimal parameters, differing by architecture and dataset, significantly affect influence estimates. Additionally, the findings suggest that optimal setting of the weight-decay factor may vary across architectures, further influencing the quality of influence estimates. This complexity indicates that multiple factors, including architecture differences and hyperparameter settings, play crucial roles in determining the accuracy of influence estimates.

## 5.2 Conclusion
This paper presents an in-depth evaluation of influence functions within the realm of deep learning, exploring their application across a range of datasets (Iris, MNIST, CIFAR-10, CIFAR-100, ImageNet) and various neural network architectures (LeNet, VGGNets, ResNets). The findings reveal that influence functions, while promising, exhibit fragility across diverse settings in deep learning. Key factors such as network depth and width, architecture, weight decay, stochastic approximations, and the choice of test points significantly impact the accuracy of influence estimates. Notably, influence estimates tend to be more reliable in simpler, shallower architectures like small CNNs and LeNet. However, accuracy declines in more complex, deeper, and wider networks such as ResNet-50, with the estimates becoming increasingly erroneous. The study also extends these evaluations to large-scale datasets like ImageNet, where it finds that influence estimates are particularly imprecise, underscoring the challenges in scaling influence function methodologies to large, complex datasets. These observations highlight the need for developing more robust influence estimators capable of navigating the intricate, non-convex landscapes characteristic of deep learning environments.

---

[^1]: Pang Wei Koh, Percy Liang: “Understanding Black-box Predictions via Influence Functions”, 2017; <a href='http://arxiv.org/abs/1703.04730'>arXiv:1703.04730</a>.

[^2]: Naman Agarwal, Brian Bullins, Elad Hazan: “Second-Order Stochastic Optimization for Machine Learning in Linear Time”, 2016, Journal of Machine Learning Research 18(116) (2017) 1-40; <a href='http://arxiv.org/abs/1602.03943'>arXiv:1602.03943</a>.

[^3]: Han Guo, Nazneen Fatema Rajani, Peter Hase, Mohit Bansal, Caiming Xiong: “FastIF: Scalable Influence Functions for Efficient Model Interpretation and Debugging”, 2020; <a href='http://arxiv.org/abs/2012.15781'>arXiv:2012.15781</a>.

[^4]: Naman Agarwal, Brian Bullins, Elad Hazan: “Second-Order Stochastic Optimization for Machine Learning in Linear Time”, 2016, Journal of Machine Learning Research 18(116) (2017) 1-40; <a href='http://arxiv.org/abs/1602.03943'>arXiv:1602.03943</a>.

[^5]: Samyadeep Basu, Philip Pope, Soheil Feizi: “Influence Functions in Deep Learning Are Fragile”, 2020; <a href='http://arxiv.org/abs/2006.14651'>arXiv:2006.14651</a>.