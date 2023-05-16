# Self-organizing a Latent Hierarchy of Sketch Patterns for Controllable Sketch Synthesis

<img src="https://github.com/CMACH508/RPCL-pix2seqH/blob/main/assets/overview.jpg" width="400" alt="overview"/>

Encoding sketches as Gaussian mixture model (GMM) distributed latent codes is an effective way to control the sketch synthesis. Each Gaussian component represents a specific sketch pattern, and a code randomly sampled from the Gaussian can be decoded to synthesize a sketch with the target pattern. However, the existing methods treat the Gaussians as the individual clusters, which neglects the relationships between them. For example, the giraffe and horse sketches heading left are related to each other by their face orientation. The relationships between sketch patterns are important messages to reveal cognitive knowledge in sketch data. Thus, it is promising to learn accurate sketch representations by modeling the pattern relationships into a latent structure. In this paper, we construct a tree-structured taxonomic hierarchy over the clusters of sketch codes. The clusters with the more specific descriptions of sketch patterns are placed at the lower levels, while the ones with the more general patterns are ranked at the higher levels. The clusters at the same rank relate to each other through inheritance of features from common ancestors. We propose a hierarchical expectation-maximization (EM)-like algorithm to explicitly learn the hierarchy, jointly with the training of the encoder-decoder network. Moreover, the learned latent hierarchy is utilized to regularize the sketch codes with the structural constraint. Experimental results show that our method significantly improves the controllable synthesis performance and obtains effective sketch analogy results.

The corresponding article was accepted by **IEEE Transactions on Neural Networks and Learning Systems** in May, 2023, and is with the authors: Sicong Zang, Shikui Tu and Lei Xu from Shanghai Jiao Tong University. The source codes and the pre-trained models will be available in the early future.

# Citation
If you find this project useful for academic purposes, please cite it as:
```
@Article{RPCL-pix2seqH,
  Title                    = {Self-organizing a Latent Hierarchy of Sketch Patterns for Controllable Sketch Synthesis},
  Author                   = {Sicong Zang and Shikui Tu and Lei Xu},
  Journal                  = {IEEE Transactions on Neural Networks and Learning Systems},
}
```
