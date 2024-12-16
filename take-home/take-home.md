# Chap1
# Chap2: Convex Optimization Basics
**Def 3: Smoothness**
- A function $f\mathrm{~is~}L\text{-smooth with respect to the }\|\cdot\|$
norm if, for any $\mathbf{x},\mathbf{y}\in\operatorname{dom}f$,
$$\|\nabla f(\mathbf{x})-\nabla f(\mathbf{y})\|_*\leq L\|\mathbf{x}-\mathbf{y}\|.$$
$||\cdot||_*$表示对偶范数.在欧几里得空间下,为$l_2$范数,上式成为:
$$\|\nabla f(\mathbf{x})-\nabla f(\mathbf{y})\|_2\leq L\|\mathbf{x}-\mathbf{y}\|_2.$$
- domain $\mathcal{X}$ 上的$\mathcal{L}-$smooth函数集合记为$C_{\mathcal{L}}^{1,1}(\mathcal{X})$.


**Lemma 1 (Descent Lemma)**
- f是凸集 $\mathcal{X}$上的L-smooth函数.则对任意 $\mathbf{x},\mathbf{y}\in\mathcal{X}$,
$$f(\mathbf{y}) \leq f(\mathbf{x})+\nabla f(\mathbf{x})^{\top}(\mathbf{y}-\mathbf{x})+\frac{L}{2}\|\mathbf{y}-\mathbf{x}\|^{2}.$$

**Definition 5: Strong Convexity.**
- A function f is σ-strongly convex if, for any $\mathbf{x},\mathbf{y}\in\operatorname{dom}f\mathrm{~and~}\lambda\in[0,1],\\f(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})\leq\lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y})-\frac\sigma2\lambda(1-\lambda)\|\mathbf{x}-\mathbf{y}\|^2.$

**Theorem 3** (*First-order Characterizations of Strong Convexity*)
- Let $f$ be a proper closed and convex function. Then for a given $\sigma > 0$, the followings are equivalent:

1. $f$ is $\sigma$-strongly convex.

2. For any $\mathbf{x} \in \operatorname{dom}(\partial f)$, $\mathbf{y} \in \operatorname{dom}(f)$ and $g \in \partial f(\mathbf{x})$,
   
   <span style="background-color: #D0E7FF; color: black; padding: 10px;">$
   f(\mathbf{y}) \geq f(\mathbf{x}) + \langle g, \mathbf{y} - \mathbf{x} \rangle + \frac{\sigma}{2} \| \mathbf{y} - \mathbf{x} \|^2.$
   *(commonly used)*

3. For any $\mathbf{x}, \mathbf{y} \in \operatorname{dom}(\partial f)$, and $g_{\mathbf{x}} \in \partial f(\mathbf{x})$, $g_{\mathbf{y}} \in \partial f(\mathbf{y})$,
   $$
   \langle g_{\mathbf{x}} - g_{\mathbf{y}}, \mathbf{x} - \mathbf{y} \rangle \geq \sigma \| \mathbf{x} - \mathbf{y} \|^2.
   $$

4. Function $f(\cdot) - \frac{\sigma}{2} \| \cdot \|^2$ is convex.
