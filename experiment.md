We provide the following experiment result to support our theoretical argument, where we show that a measurement-consistent image that is obtained without explicitly enforcing backward consistency can not be represented as the output of $f(v_t;t,\epsilon_\theta)$ with some $v_t$. This is done by examining whether the following loss can go to zero or not.

$\min_z L(z) := \|\hat{x}'(x_t) - f(z; t, \epsilon_\theta)\|^2_2$
 

with $z$ initialized as $x_t$.
Here, $\hat{x}'_0(x_t)$ is the measurement-consistent vector that is obtained **without enforcing our backward consistency** explicitly. Given some $x_t$, it is obtained by:
1. $\hat{x}(x_t) = f(x_t;t,\epsilon_\theta)$
2. $\hat{x}'_0(x_t) = \arg\min_x \|A(x)-y\|^2_2$ with $x$ initialized as $\hat{x}_0(x_t)$
In the following link (https://anonymous.4open.science/r/SITCOM-B65F/loss_plot.png), we plot the average $L(z)$ over optimization iteration with indicating the minimum and maximum. This is averaged over 20 images for the non-linear deblurring task.
As observed, the loss value can not go to zero which means that no $z$ (i.e. $v_t$) was found such that $\hat{x}'_0(x_t)$ satisfies the backward consistency. We acknowledge that this could be due to the optimization becoming stuck in a local region of the highly non-linear landscape, limiting the search to a neighborhood of the initial point. Therefore, we interpret this result as partial empirical evidence supporting our theoretical argument, rather than a rigorous proof.
