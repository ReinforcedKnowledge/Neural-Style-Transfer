# Introduction
The above code follows and offers an implementation for the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys et al.) [1].
The goal of the work is to recreate an image by using the style of another image (e.g. a painting), as we can see below we recreate a photo of New York (taken from [iloveny.com](https://www.iloveny.com/)) using the style of [Van Gogh](https://en.wikipedia.org/wiki/Vincent_van_Gogh) extracted from its [The Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night) painting.

# How It Works
To recreate an image using the style extracted from another image we have to understand first what is the *style*. That's why the authors distinguish between the *content* of an image, and its style. The content of an image will describe what the image is made of, its components etc. while the style of an image will describe the texture information of an image. It is a restricted definition of what we understand intuitively and the shortcomings will be discussed in a later section.

To formalize these two notions the authors use [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network). By observing the activations of a Convolutional Neural Newtork we notice that the first layers extract elementary caracteristics such as lines, edges etc. while deeper layers extract higher caracteristics such as the presence of a tire etc. [2]

So the content of an image could be synthesized using the activations of deep layers of a CNN after getting the image through the forward pass, but these layers don't preserve the exact location of the object they extract such as if our initial image contained a dog at the center, the activations would certainly return a dog but not necessarily at the center. Meanwhile, the firsts layers of a CNN will preserve the location of pixels. **We need a trade-off between the firsts and the deepest layers**.

Let $l$ denote the index of a convolutional layer and let $N_{l}$ denote the number of filters it uses to produce a volume of feature map of size $(W_{l}, H_{l}, N_{l})$ with $(W_{l}, H_{l})$ the shape of a feature map. We can represent that volume by a matrix $a^{l}$ in $M_{N_{l}, W_{l} \times H_{l}}(R)$. $a_{i,.}^{l}$ is the vector obtained by unrolling the $i$-th feature map, and that's why it contains $W_{l} \times H_{l}$ and that there are $N_{l}$ rows.

Now if we have a generated image **G** for which we want to paste the *content* of the content image **C**, we can change the pixels of **G** such as the different convolutional layers' activations while passing **G** are as close as possible to those we get by passing **C** through the CNN. Formally, if we denote the activations of layer $l$ when we pass **C** by $a^{l, [C]}$ and the activations of the same layer when we pass **G** by $a^{l, [G]}$ then by minimizing the **content loss** $L_{content} = \lVert a^{l, [C]} - a^{l, [G]} \rVert _{F}^{2}$, **G would resemble C**. Here $\lVert A \rVert _{F}$ represents the [Frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html).

# References
[1] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

[2] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." In European conference on computer vision, pp. 818-833. Springer, Cham, 2014.

[3] Gatys, Leon, Alexander S. Ecker, and Matthias Bethge. "Texture synthesis using convolutional neural networks." _Advances in neural information processing systems_ 28 (2015).
