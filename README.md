# Introduction
The above code follows and offers an implementation for the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys et al.) [1].
The goal of the work is to recreate an image by using the style of another image (e.g. a painting), as we can see below we recreate a photo of New York (taken from [iloveny.com](https://www.iloveny.com/)) using the style of [Van Gogh](https://en.wikipedia.org/wiki/Vincent_van_Gogh) extracted from its [The Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night) painting.

![title](https://github.com/ReinforcedKnowledge/Neural-Style-Transfer/blob/main/images/mixed_new_york.png)

# How It Works
To recreate an image using the style extracted from another image we have to understand first what is the *style*. That's why the authors distinguish between the *content* of an image, and its style. The content of an image will describe what the image is made of, its components etc. while the style of an image will describe the texture information of an image. It is a restricted definition of what we understand intuitively and the shortcomings will be discussed in a later section.

To formalize these two notions the authors use [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network). By observing the activations of a Convolutional Neural Newtork we notice that the first layers extract elementary caracteristics such as lines, edges etc. while deeper layers extract higher caracteristics such as the presence of a tire etc. [2]

So the content of an image could be synthesized using the activations of deep layers of a CNN after getting the image through the forward pass, but these layers don't preserve the exact location of the object they extract such as if our initial image contained a dog at the center, the activations would certainly return a dog but not necessarily at the center. Meanwhile, the firsts layers of a CNN will preserve the location of pixels. **We need a trade-off between the firsts and the deepest layers**.

Let $l$ denote the index of a convolutional layer and let $N_{l}$ denote the number of filters it uses to produce a volume of feature map of size $(W_{l}, H_{l}, N_{l})$ with $(W_{l}, H_{l})$ the shape of a feature map. We can represent that volume by a matrix $a^{l}$ in $M_{N_{l}, W_{l} \times H_{l}}(R)$. $a_{i,.}^{l}$ is the vector obtained by unrolling the $i$-th feature map, and that's why it contains $W_{l} \times H_{l}$ and that there are $N_{l}$ rows.

Now if we have a generated image **G** for which we want to paste the *content* of the content image **C**, we can change the pixels of **G** such as the different convolutional layers' activations while passing **G** are as close as possible to those we get by passing **C** through the CNN. Formally, if we denote the activations of layer $l$ when we pass **C** by $a^{l, [C]}$ and the activations of the same layer when we pass **G** by $a^{l, [G]}$ then by minimizing the **content loss** $L_{content} = \lVert a^{l, [C]} - a^{l, [G]} \rVert _{F}^{2}$, **G would resemble C**. Here $\lVert A \rVert _{F}$ represents the [Frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html).

The explicit formula of the loss is then $L_{content} = \frac{1}{2} \sum_{i, j}{(a_{i,j}^{l, [C]} - a_{i,j}^{l, [G]})^{2}}$. The derivative of this loss with respect to the activations is $\frac{\partial L_{content}}{\partial a_{i,j}^{l, [G]}} = a_{i,j}^{l, [G]} (a_{i,j}^{l, [G]} - a_{i,j}^{l, [C]})$. With back-propagation (done by TensorFlow as an auto-differentiation framework), gradient with respect to the pixels of the generated image **G** and then be updated to minimize the loss.

To get the texture of an image we use the correlations between differents feature maps of a same activation volume of a convolutional layer $l$ [3]. We see the style as a set of caracteristics that come together as using horizontal lines with blue colors to paint the sky etc. That's why we need to correlations between the feature maps to capture what *defines* the style. Formally, if we have two matrices $A$ and $B$ of same shape $(m, n)$ then if we denote the vector we get by unrolling $A$ and $B$ into $a$ and $b$ respectively, the correlation between $A$ and $B$ is $\frac{a \cdot b}{Var(A)Var(B)}$.

Now, if our activation volume is denoted by $a^{l}$, we can compute the Gram matrix of $a^{l}$ by taking the correlations between each two feature maps of this volume and by not dividing by the product of variances. So if $a_{i,.}^{l}$ and $a_{j,.}^{l}$ are the $i$-th and $j$-th feature maps respectively, the $i, j$-th element of the Gram matrix would be the correlation between these two feature maps, which is $G_{i,j}^{l} = \sum_{k}{a_{i,k}^{l}a_{j,k}^{l}}$. One of the advantages of the Gram matrix is that it doesn't care about the location of the features. Notice that the Gram matrix is symmetric, which we'll use later in computing the derivative of the style loss with respect to the activations of the layer.
 
 By using different layers $l$ we obtain a more stable representation of the *style* and also a more complete one. So to compare the styles of a style image **S** and the generated image **G** we take the sum of the squared Frobenius norm of the difference of Gram matrices through different layers. Formally, for a specific layer $l$, the loss normalized loss is $E_{l} = \frac{1}{4}\frac{1}{(W_{ł}H_{l}N_{l})^{2}}\sum_{i,j}(G_{generated,i,j}^{l}-G_{style,i,j}^{l})^{2}$. We normalize by $W_{ł}H_{l}N_{l}$ since we take the norm squared. The $\frac{1}{4}$ is there only to get a simpler derivative formula.

The derivative of this loss with respect to the activations is $\frac{\partial E_{ł}}{\partial a_{i,j}^{l,[G]​}}$. Since only $G_{generated,i,j}^{l}$ and $G_{generated,j,i}^{l}$ contain $a_{i,j}^{l,[G]​}$, we only care about them. $\frac{\partial G_{generated,i,j}^{l}}{\partial a_{i,j}^{l,[G]​}} = \frac{\partial}{\partial a_{i,j}^{l,[G]​}} \sum_{k}{a_{i,k}^{l,[G]}a_{j,k}^{l,[G]}} = a_{j,i}^{l,[G]}$. Using the symmetric of the Gram matrices we get $\frac{\partial E_{ł}}{\partial a_{i,j}^{l,[G]​}} = \frac{1}{(W_{l}H_{l}N_{l})^{2}}a_{j,i}^{l,[G]}(G_{generated,j,i}^{l} - G_{style,i,j}^{l})$. Gradient of $E_{l}$ with activations in lower layers can be computed using back-propagation until we computed gradients with respect to the pixels of the generated image.

As we said, we take many layers, hence the final **style loss** is $L_{style}=\sum_{l}w_{l}E_{l}$. The weights $w_{l}$ are used to give importance to some layers rather than others. In the image in the introduction we used uniform weights of $0.25$.

The final image **G** is then synthesized by matching the contents of the content image **C** using the style of the style image **C**. We want the location and global arrangement of the contents to be preserved while the colours and local structures and texture to be taken from the style image. That's why the final loss that's computed is $L(G, C, S) = \alpha L_{content}(G, C) + \beta L_{style}(G, S)$. $\alpha$ and $\beta$ control the value of preserving the content and style respectively. A high $\frac{\alpha}{\beta}$ ratio value would give more importance to preserving the content and have it less distorted while a small value of the ratio would give more importance to the style and would result into a more texturized image.

# References
[1] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

[2] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." In European conference on computer vision, pp. 818-833. Springer, Cham, 2014.

[3] Gatys, Leon, Alexander S. Ecker, and Matthias Bethge. "Texture synthesis using convolutional neural networks." _Advances in neural information processing systems_ 28 (2015).
