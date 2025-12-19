# Theory of Neural Dimensionality Estimation

This library provides a framework for estimating the **Intrinsic Dimensionality (ID)** of a dataset or the **Interaction Dimensionality** between two datasets using Neural Mutual Information Estimation.

## The Core Hypothesis: Information Saturation

Traditional dimensionality estimation methods (like PCA) rely on variance or linear projections. This library relies on **Information Theory**.

The core hypothesis is as follows:

Let $X$ and $Y$ be two high-dimensional variables with some shared dependency. We use encoders $f(\cdot)$ and $g(\cdot)$ that map these variables to a lower-dimensional embedding space $Z$ of dimension $k_z$:
$$Z_X = f(X), \quad Z_Y = g(Y), \quad \text{where } Z \in \mathbb{R}^{k_z}$$

We define the estimated Mutual Information for a specific bottleneck dimension $k_z$ as:
$$\hat{I}_{k_z} = \max_{f, g} I(Z_X; Z_Y)$$

As we sweep the embedding dimension $k_z$ from 1 to $D$:
1.  **Growth:** Initially, as $k_z$ increases, the encoders can capture more of the shared dependency, so $\hat{I}_{k_z}$ rises.
2.  **Saturation:** Once $k_z$ reaches the true **Intrinsic Dimensionality (ID)** of the interaction, adding more dimensions to $Z$ adds no new shared information. $\hat{I}_{k_z}$ saturates (plateaus).

**The Estimator:** The ID is defined as the smallest $k_z$ where the Mutual Information saturates.

> **Note:** We are less concerned with the absolute value of MI (which might be the target, and in fact, it's a byproduct of correct estimation) and more concerned with the **trend** and the **saturation point**.

## Scenarios

### 1. Two Datasets: Interaction Dimensionality
Given two distinct views $X$ and $Y$ (e.g., Audio and Video of the same event, or two biological modalities):
* We sweep $k_z$.
* The saturation point tells us the number of independent latent factors coupling $X$ and $Y$.

### 2. Single Dataset: Intrinsic Dimensionality
Given a single dataset $X$ (e.g., a collection of images):
* We must artificially construct a pair $(X', Y')$.
* **Strategies:**
    * **Splitting:** $X'$ is the left half of an image, $Y'$ is the right half.
    * **Temporal:** $X'$ is frame(s) $t$, $Y'$ is frame(s) $t + \text{lag}$.
    * **Augmentation:** $X'$ is an augmented view (crop), $Y'$ is a different view (color jitter).
* The saturation point of $I(Z_{X'}; Z_{Y'})$ estimates the intrinsic dimensionality of the manifold $X$ lives on.

## The Estimator Challenge

### The Neural Bounds (InfoNCE)
Directly computing $I(Z_X; Z_Y)$ is intractable. We use neural variational lower bounds, primarily the **InfoNCE** (Noise Contrastive Estimation) bound, often parametrized as:
$$I(Z_X; Z_Y) \geq \mathbb{E}\left[ \frac{1}{K} \sum_{i=1}^K \log \frac{e^{h(z_{x_i}, z_{y_i})}}{\frac{1}{K} \sum_{j=1}^K e^{h(z_{x_i}, z_{y_j})}} \right]$$
where $h(\cdot, \cdot)$ is a "Critic" function.

### The Critic Problem & The Hybrid Solution
To find the true saturation point, our estimator must be "capacity tight"—it must be able to capture all available information at a given $k_z$.

* **Separable Critics (Inner Product):** $h(z_x, z_y) = z_x^T z_y$.
    * *Pros:* Fast, mathematically clean.
    * *Cons:* Restrictive geometry. For complex distributions, a separable critic might fail to capture all information even if $k_z$ is sufficient. This leads to **False Saturation** (overestimating ID).
* **The Hybrid Critic (Novelty):**
    * We use an MLP *after* the projection: $h(z_x, z_y) = \text{MLP}([z_x, z_y])$.
    * This ensures that if information exists in the $k_z$-dimensional embeddings, the critic has enough capacity to find it. This provides a robust "Upper Bound of the Lower Bound," ensuring the saturation we see is due to the data, not the model's limitations.

## Future Integrations
While this library focuses on Neural Estimation, intrinsic dimensionality is a broad field. Future versions will integrate classical estimators for comparison:
* Maximum Likelihood Estimation (Levina-Bickel)
* Two-Nearest Neighbors (TwoNN)