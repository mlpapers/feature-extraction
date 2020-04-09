# Feature extraction
> In machine learning, pattern recognition and in image processing, feature extraction starts from an initial set of measured data and builds derived values (features) intended to be informative and non-redundant, facilitating the subsequent learning and generalization steps, and in some cases leading to better human interpretations. Feature extraction is related to dimensionality reduction. ([Wiki](https://en.wikipedia.org/wiki/Feature_extraction))

- **Overview**
  - [A survey of dimensionality reduction techniques](https://arxiv.org/pdf/1403.2877.pdf)
  - [Feature Selection and Feature Extraction in Pattern Analysis: A Literature Review](https://arxiv.org/pdf/1905.02845.pdf) (2019) *Benyamin Ghojogh, Maria N. Samad, Sayema Asif Mashhadi,Tania Kapoor, Wahab Ali, Fakhri Karray, Mark Crowley*

- **PCA** Principal Component Analysis ([Wiki](https://en.wikipedia.org/wiki/Principal_component_analysis))
  - [On lines and planes of closest fit to systems of points in space](https://zenodo.org/record/1430636#.Xos47PFRVnx) (1901) *Karl Pearson*
  - Supervised PCA: [Prediction by Supervised Principal Components](https://web.stanford.edu/~hastie/Papers/spca_JASA.pdf) (2006) *Eric Bair, Trevor Hastie, Debashis Paul, Robert Tibshirani*
  - Dual Principal Component Analysis
  - Kernel Principal Component Analysis
- **ICA** Independent Component Analysis ([Wiki](https://en.wikipedia.org/wiki/Independent_component_analysis))
- **FLDA** Fisher Linear Discriminant Analysis  
  > Similar to PCA, FLDA calculates the projection of data along a direction; however, rather than maximizing the variation of data, FLDA utilizes label information to get a projection maximizing the ratio of between-class variance to within-class variance. ([Source](https://arxiv.org/pdf/1905.02845.pdf))
  - Supervised
- **KFLDA** Kernel Fisher Linear Discriminant Analysis
- **Factor analysys**  
  > This technique is used to reduce a large number of variables into fewer numbers of factors. The values of observed data are expressed as functions of a number of possible causes in order to find which are the most important. The observations are assumed to be caused by a linear transformation of lower-dimensional latent factors and added Gaussian noise. ([Source](https://towardsdatascience.com/dimensionality-reduction-101-for-dummies-like-me-abcfb2551794))
- **t-SNE** ([Homepage](https://lvdmaaten.github.io/tsne/), [Wiki](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding), [CRAN](https://cran.r-project.org/web/packages/tsne/), [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html))
  - [Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf) (2008) *Laurens van der Maaten, Geoffrey Hinton*
  - [Accelerating t-SNE using Tree-Based Algorithms](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) (2014) *Laurens van der Maaten*
  - **Tree-SNE** - Hieararchical t-SNE  ([Code](https://github.com/isaacrob/treesne))
    - [Tree-SNE: Hierarchical Clustering and Visualization Using t-SNE](https://arxiv.org/pdf/2002.05687) (2020) *Isaac Robinson, Emma Pierce-Hoffman*
  - **Let-SNE**
    - [Let-SNE: A Hybrid Approach to Data Embedding and Visualization of Hyperspectral Imagery](https://arxiv.org/pdf/1910.08790.pdf) (2020) *Megh Shukla, Biplab Banerjee, Krishna Mohan Buddhiraju*
- **Kernel PCA**
- **LLE** Locally Linear Embedding  
  > Constructs a k-nearest neighbor graph similar to Isomap. Then it tries to locally represent every data sample x i using a weighted summation of its k-nearest neighbors. ([Source](https://arxiv.org/pdf/1905.02845.pdf))
- **HLLE** Hessian Eigenmapping  
  > Projects data to a lower dimension while preserving the local neighborhood like LLE but uses the Hessian operator to better achieve this result and hence the name. ([Source](https://towardsdatascience.com/dimensionality-reduction-for-machine-learning-80a46c2ebb7e))
- **Laplacian Eigenmap** Spectral Embedding
- **Maximum Variance Unfolding**
- **NMF** Non-negative matrix factorization
- **Isomap**
- **UMAP** Uniform Manifold Approximation and Projection ([Code](https://github.com/lmcinnes/umap), ([GPU version](https://docs.rapids.ai/api/cuml/stable/api.html#umap))
- **Trimap** ([Code](https://github.com/eamid/trimap), [PyPI](https://pypi.org/project/trimap/))
  - [Trimap: Large-scale Dimensionality Reduction Using Triplets](https://arxiv.org/pdf/1910.00204.pdf) (2019) *Ehsan Amid, Manfred K. Warmuth*
- **Autoencoders** ([Wiki](https://en.wikipedia.org/wiki/Autoencoder))
- **SOM** Self-Organizing Maps or Kohonen Maps ([Wiki](https://en.wikipedia.org/wiki/Self-organizing_map))
  - [Self-Organized Formation of Topologically Correct Feature Maps](http://www.cnbc.cmu.edu/~tai/nc19journalclubs/Kohonen1982_Article_Self-organizedFormationOfTopol.pdf) (1982) *Teuvo Kohonen*
- **Sammon’s Mapping**
- **SDE** Semi-definite embedding
- **LargeVis**
  - [Visualizing Large-scale and High-dimensional Data](https://arxiv.org/abs/1602.00370) (2016) *Jian Tang, Jingzhou Liu, Ming Zhang, Qiaozhu Mei*
