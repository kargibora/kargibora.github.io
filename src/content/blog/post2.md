---
title: "Influence Functions in XAI"
description: ""
pubDate: "Jan 15 2024"
heroImage: "https://files.oaiusercontent.com/file-qbMQMDwXIkaI9aye5OLAnzgS?se=2024-01-15T21%3A59%3A37Z&sp=r&sv=2021-08-06&sr=b&rscc=max-age%3D31536000%2C%20immutable&rscd=attachment%3B%20filename%3Dadab192d-189d-4ca4-a2f5-8c3692f442a3.webp&sig=vUkc08%2BZucdsybnE/lZHo4t25sFuI%2BK27z5upOhlXAk%3D"
tags: ["XAI","Paper Review"]
layout: '../../layouts/MarkdownLayout.astro'
badge: ""
timeToRead: "15"
---

# 1. Introduction
In the ever-evolving field of deep learning, the need for understanding the model decision becomes more important each day. To fully incorprate deep learning systems into a critical decision systems such as self-driving cars and medical AI supervisors, tracing back the algorithm into its' data point gives us a lot of understanding about the decision system itself. However deep learning systems are *often* blackboxes. 

One simple question arises from these concerns:
$$
\textnormal{"What is the effect of a single data point on the decision?"}
$$

One can find how *influential* a data point $x$ is on the decision of the model $f_\theta(x^*)$ where $x^*$ is a point not seen in the dataset by asking:
$$
\textnormal{"What would happen if the training point was not in the dataset; or was perturbed a little bit?}
$$

A simple way to calculate this effect is to retrain your model *without the data point* $x$: however this is a very costly approach as current deep learning models usually takes a long time to train.

Koh et Al. [^1] use *influence functions*, which is a powerful tool in statistics, to overcome this problem. Instead of *retraining approach*, they compute how *influential* a point is on the decision with the *influential functions*.

# 2. Influence Functions
Consider a function, which takes input from the input space $\mathcal X$, to an output space $\mathcal Y$ given the training points $(z_i,...,z_n)$ where $z_i = (x_i,y_i) \in \mathcal X \times \mathcal Y$. Define an arbitrary lose function $L$ defined with the model parameters $\theta \in \Theta$ where
$$
L(z,\theta) = \frac{1}{n}\sum_{i=1}^n L(z_i,\theta) \tag{2.1}
$$

The paper also assumes about the *twice-differentiability* and *convexity* of the $\theta$. This will be important later on. Formally defining the influence of a point $x$, we have to define the change in model parameters we get after removing a point as:
$$
\hat \theta_{- z} - \hat \theta \quad \textnormal{where} \quad \hat \theta_{-z} = \argmin_{\theta \in \Theta}\sum_{z_i \neq z}L(z_i,\theta)  \tag{2.2}
$$
where $z = (x,y)$ is the point we are removing, and $\hat \theta$ are the model parameters that minimizes the empirical risk $R(\theta) = \frac{1}{n}\sum_{i=1}^n L(z_i,\theta)$.

Instead of retraining to get $\hat \theta_{-z}$, what we if we approximate the result by checking the sensivity of a point $z$ to a weight in the loss? To be more formally, given $\epsilon$, define:
$$
\hat \theta_{\epsilon, z} = \argmin_{\theta \in \Theta} \frac{1}{n} \sum_{i=1}^{n}L(z_i, \theta) + \epsilon L(z,\theta)  \tag{2.3}
$$
Notice that this is basically the model parameters after training the model with added weight to the $z$. Influence of upweighting $z$ can be calculated as:
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
Since $\epsilon \rightarrow 0$ and $\nabla R(\hat \theta) = 0$, as $\hat \nabla$ minimizes the risk function $R(\theta)$, we can write the equation $(2.6)$ as 
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
= -H_{\hat \theta}^{-1}\nabla L(z,\hat \theta)
\end{align*} \tag{2.9}
$$

Now the effect of upweighting the point $z$ is derived, let us calculate the effect of upweighting on the *loss function* by using the chain rule:
$$
\begin{align*}
\mathcal I_{up,loss}(z,z_{test}) = \frac{dL(z_{test}, \hat \theta_{\epsilon,z})}{d\epsilon} \big |_{\epsilon=0} \\
= \nabla_\theta L(z_{test}, \hat \theta)^T \frac{d\hat \theta_{\epsilon,z}}{d\epsilon} \big |_{\epsilon=0} \\
= -\nabla_\theta L(z_{test}, \hat \theta)^T H_{\hat \theta}^{-1}\nabla L(z,\hat \theta)
\end{align*} \tag{2.10}
$$
where ${z_{test}}$ is the test point and $z$ is the point that is *upweighted*. Basically, $\mathcal I_{up,loss}(z,z_{test})$ approximates the *influence of the point* $z$ *on the* $z_{test}$.

However the problem of this approach is, the inverse of Hessian is quite expensive to compute. This bring us to the next section.

## 2.1 Efficiently calculating the influence

Inverting the Hessian of the empirical risk takes $O(np^2 + p^3)$ operations where $n$ is the number of points and $p$ is the dimension of the weights $\theta \in \mathbb R^p$. To efficiently compute the inverse of a Hessian, authors use Hessian-vector products (HVPs) to efficiently approximate:
$$
s_{test} = H_{\hat \theta}^{-1} \nabla_\theta L(z_{test}, \hat \theta)
$$

Authors discuss two algorithms in order to efficiently compute Hessian approximations.

### 2.1.1 Conjugate Gradients

### 2.1.2 Stochastic Estimation


### 3. Experiments & Results

### 4. Faster influence functions with FastIF

Even after the important speedups, computing influence functions in usual ways is slow due to the dimensionality of the data. *FastIF* [^3] introduces a speed-up to the conventional way with the 


---

[^1]: Pang Wei Koh, Percy Liang: “Understanding Black-box Predictions via Influence Functions”, 2017; <a href='http://arxiv.org/abs/1703.04730'>arXiv:1703.04730</a>.

[^2]: Naman Agarwal, Brian Bullins, Elad Hazan: “Second-Order Stochastic Optimization for Machine Learning in Linear Time”, 2016, Journal of Machine Learning Research 18(116) (2017) 1-40; <a href='http://arxiv.org/abs/1602.03943'>arXiv:1602.03943</a>.

[^3]: Han Guo, Nazneen Fatema Rajani, Peter Hase, Mohit Bansal, Caiming Xiong: “FastIF: Scalable Influence Functions for Efficient Model Interpretation and Debugging”, 2020; <a href='http://arxiv.org/abs/2012.15781'>arXiv:2012.15781</a>.