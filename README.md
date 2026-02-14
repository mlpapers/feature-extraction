# Feature extraction
> In machine learning, pattern recognition and in image processing, feature extraction starts from an initial set of measured data and builds derived values (features) intended to be informative and non-redundant, facilitating the subsequent learning and generalization steps, and in some cases leading to better human interpretations. Feature extraction is related to dimensionality reduction. ([Wiki](https://en.wikipedia.org/wiki/Feature_extraction))

- **Overview**
  - [A survey of dimensionality reduction techniques](https://arxiv.org/pdf/1403.2877.pdf) *C.O.S.Sorzano, J.Vargas, A.Pascual‐Montano*
  - [Feature Selection and Feature Extraction in Pattern Analysis: A Literature Review](https://arxiv.org/pdf/1905.02845.pdf) (2019) *Benyamin Ghojogh, Maria N. Samad, Sayema Asif Mashhadi,Tania Kapoor, Wahab Ali, Fakhri Karray, Mark Crowley*

- **PCA** Principal Component Analysis ([Wiki](https://en.wikipedia.org/wiki/Principal_component_analysis))
  - [On lines and planes of closest fit to systems of points in space](https://zenodo.org/record/1430636#.Xos47PFRVnx) (1901) *Karl Pearson*
  - Supervised PCA: [Prediction by Supervised Principal Components](https://web.stanford.edu/~hastie/Papers/spca_JASA.pdf) (2006) *Eric Bair, Trevor Hastie, Debashis Paul, Robert Tibshirani*
  - Sparse PCA ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA))
- **DPCA** Dual Principal Component Analysis
- **KPCA** Kernel Principal Component Analysis ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA), [Wiki](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis))
  - [Nonlinear Component Analysis as a Kernel Eigenvalue Problem](http://alex.smola.org/papers/1998/SchSmoMul98.pdf) (1998) *Bernhard Scholkopf, Alexander Smola, Klaus-Robert Muller*
  - [Kernel PCA for Novelty Detection](http://www.heikohoffmann.de/documents/hoffmann_kpca_preprint.pdf) (2006) *Heiko Hoffmann*
  - [Robust Kernel Principal Component Analysis](https://papers.nips.cc/paper/3566-robust-kernel-principal-component-analysis.pdf) *Minh Hoai Nguyen, Fernando De la Torre*
- **IPCA** Incremental (online) PCA ([CRAN](https://cran.r-project.org/web/packages/onlinePCA/), [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA))
- **ICA** Independent Component Analysis ([Wiki](https://en.wikipedia.org/wiki/Independent_component_analysis))
  - [Independent Component Analysis: Algorithms and Applications](http://mlsp.cs.cmu.edu/courses/fall2012/lectures/ICA_Hyvarinen.pdf) (2000) *Aapo Hyvärinen, Erkki Oja*
  - [Independent Component Analysis](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf) (2001) - Free ebook *Aapo Hyvarinen, Juha Karhunen, Erkki Oja*
  - FastICA ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA))
- **FLDA** Fisher's Linear Discriminant Analysis (Supervised) ([Wiki](https://en.wikipedia.org/wiki/Linear_discriminant_analysis))  
  > Similar to PCA, FLDA calculates the projection of data along a direction; however, rather than maximizing the variation of data, FLDA utilizes label information to get a projection maximizing the ratio of between-class variance to within-class variance. ([Source](https://arxiv.org/pdf/1905.02845.pdf))
  - [The Use of Multiple Measurements in Taxonomic Problems](https://digital.library.adelaide.edu.au/dspace/bitstream/2440/15227/1/138.pdf) (1936) *R. A. Fisher*
  - [The Utilization of Multiple Measurements in Problems of Biological Classification](https://www.jstor.org/stable/2983775?seq=1) (1948) - require registration *C. Radhakrishna Rao*
  - [PCA versus LDA](http://www2.ece.ohio-state.edu/~aleix/pami01.pdf) (2001) *Aleix M. Martinez, Avinash C. Kak*
  - Package: MASS includes lda ([CRAN](https://cran.r-project.org/web/packages/MASS/))
  - Package: sda ([CRAN](https://cran.r-project.org/web/packages/sda/index.html))
- **KFLDA** Kernel Fisher Linear Discriminant Analysis
- **MDS** Multidimensional Scaling ([Wiki](https://en.wikipedia.org/wiki/Multidimensional_scaling))
  - [Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis](http://cda.psych.uiuc.edu/psychometrika_highly_cited_articles/kruskal_1964a.pdf) (1964) *J. B. Kruskal*
  - [An Analysis of Classical Multidimensional Scaling](https://arxiv.org/pdf/1812.11954.pdf) (2019) *Anna Little, Yuying Xie, Qiang Sun*
  - Packages:
      [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)
- **Isomap** ([Homepage](https://web.mit.edu/cocosci/isomap/isomap.html), [Wiki](https://en.wikipedia.org/wiki/Isomap))
  - [A Global Geometric Framework for Nonlinear Dimensionality Reduction](https://web.mit.edu/cocosci/Papers/sci_reprint.pdf) (2000) *Joshua B. Tenenbaum, Vin de Silva, John C. Langford*
  - Packages:
      [dimRed](https://cran.r-project.org/web/packages/dimRed/dimRed.pdf),
      [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html)
- **Latent Dirichlet Allocation**
  - [Online Learning for Latent Dirichlet Allocation](https://www.di.ens.fr/~fbach/mdhnips2010.pdf) (2010) *Matthew D. Hoffman, David M. Blei, Francis Bach*
- **Factor analysys** ([Wiki](https://en.wikipedia.org/wiki/Factor_analysis), [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis))  
  > This technique is used to reduce a large number of variables into fewer numbers of factors. The values of observed data are expressed as functions of a number of possible causes in order to find which are the most important. The observations are assumed to be caused by a linear transformation of lower-dimensional latent factors and added Gaussian noise. ([Source](https://towardsdatascience.com/dimensionality-reduction-101-for-dummies-like-me-abcfb2551794))
- **t-SNE** ([Homepage](https://lvdmaaten.github.io/tsne/), [Wiki](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding), [CRAN](https://cran.r-project.org/web/packages/tsne/), [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html))
  - [Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf) (2008) *Laurens van der Maaten, Geoffrey Hinton*
  - [Accelerating t-SNE using Tree-Based Algorithms](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) (2014) *Laurens van der Maaten*
  - **Tree-SNE** - Hieararchical t-SNE  ([Code](https://github.com/isaacrob/treesne))
    - [Tree-SNE: Hierarchical Clustering and Visualization Using t-SNE](https://arxiv.org/pdf/2002.05687) (2020) *Isaac Robinson, Emma Pierce-Hoffman*
  - **Let-SNE**
    - [Let-SNE: A Hybrid Approach to Data Embedding and Visualization of Hyperspectral Imagery](https://arxiv.org/pdf/1910.08790.pdf) (2020) *Megh Shukla, Biplab Banerjee, Krishna Mohan Buddhiraju*
- **LLE** Locally Linear Embedding  
  > Constructs a k-nearest neighbor graph similar to Isomap. Then it tries to locally represent every data sample x i using a weighted summation of its k-nearest neighbors. ([Source](https://arxiv.org/pdf/1905.02845.pdf))
- **HLLE** Hessian Eigenmapping  
  > Projects data to a lower dimension while preserving the local neighborhood like LLE but uses the Hessian operator to better achieve this result and hence the name. ([Source](https://towardsdatascience.com/dimensionality-reduction-for-machine-learning-80a46c2ebb7e))
- **Laplacian Eigenmap** Spectral Embedding
- **Maximum Variance Unfolding**
- **NMF** Non-negative matrix factorization
- **UMAP** Uniform Manifold Approximation and Projection ([Code](https://github.com/lmcinnes/umap), [GPU version](https://docs.rapids.ai/api/cuml/stable/api.html#umap))
  - [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/pdf/1802.03426) (2018) *Leland McInnes, John Healy, James Melville*
- **Trimap** ([Code](https://github.com/eamid/trimap), [PyPI](https://pypi.org/project/trimap/))
  - [Trimap: Large-scale Dimensionality Reduction Using Triplets](https://arxiv.org/pdf/1910.00204.pdf) (2019) *Ehsan Amid, Manfred K. Warmuth*
- **Autoencoders** ([Wiki](https://en.wikipedia.org/wiki/Autoencoder))
- **SOM** Self-Organizing Maps or Kohonen Maps ([Wiki](https://en.wikipedia.org/wiki/Self-organizing_map))
  - [Self-Organized Formation of Topologically Correct Feature Maps](http://www.cnbc.cmu.edu/~tai/nc19journalclubs/Kohonen1982_Article_Self-organizedFormationOfTopol.pdf) (1982) *Teuvo Kohonen*
- **Sammon’s Mapping**
- **SDE** Semi-definite embedding
- **LargeVis**
  - [Visualizing Large-scale and High-dimensional Data](https://arxiv.org/abs/1602.00370) (2016) *Jian Tang, Jingzhou Liu, Ming Zhang, Qiaozhu Mei*

## Software
- **R**
  - dimRed ([CRAN](https://cran.r-project.org/web/packages/dimRed/))
  - dyndimred ([CRAN](https://cran.r-project.org/web/packages/dyndimred/))
  - intrinsicDimemsion ([CRAN](https://cran.r-project.org/web/packages/intrinsicDimension/))
  - Rdimtools ([Paper](https://arxiv.org/pdf/2005.11107.pdf), [CRAN](https://cran.r-project.org/web/packages/Rdimtools/))
- **Python**
  - scikit-learn
  - umap-learn ([Homepage](https://umap-learn.readthedocs.io), [PyPI](https://pypi.org/project/umap-learn/))
- **Javascript**
  - tsne ([NPM](https://www.npmjs.com/package/tsne))
  - umap-js ([NPM](https://www.npmjs.com/package/umap-js))
  - dimred ([NPM](https://www.npmjs.com/package/dimred))
- **C++**
  - tapkee ([Code](https://github.com/lisitsyn/tapkee))
- **Web**
  - StatSim ([Vis](https://statsim.com/vis/))

## Related Topics
- [Feature Selection](https://mlpapers.org/feature-selection/)
- [Neural Networks](https://mlpapers.org/neural-nets/)
- [Multiview Learning](https://mlpapers.org/multiview-learning/)
- [Clustering](https://mlpapers.org/clustering/)
