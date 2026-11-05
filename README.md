# Feature Extraction and Representation Learning

🔒 Full text at this link requires payment or institutional access. Unmarked links may be unchecked.

## Surveys

- **Representation Learning: A Review and New Perspectives** - Reviews probabilistic models, autoencoders and manifold learning as approaches to learning representations. [Bengio et al. 2013](<https://arxiv.org/abs/1206.5538> "Representation Learning: A Review and New Perspectives; TPAMI").
- **A Survey of Dimensionality Reduction Techniques** - Categorizes dimensionality-reduction techniques and explains their mathematical foundations. [Sorzano et al. 2014](<https://arxiv.org/abs/1403.2877> "A survey of dimensionality reduction techniques; arXiv").
- **Feature Selection and Feature Extraction in Pattern Analysis** - Reviews the theory, applications, and numerical implementations of feature selection and extraction methods. [Ghojogh et al. 2019](<https://arxiv.org/abs/1905.02845> "Feature Selection and Feature Extraction in Pattern Analysis: A Literature Review; arXiv").

## Fixed and domain-informed features

### Text and image descriptors

- **Text vectorization** - Sparse word/character n-grams, TF-IDF, and feature hashing. [docs](<https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction>).
- **Image descriptors** - SIFT describes local keypoints; HOG summarizes local gradient orientations. [Lowe 2004](<https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf> "Distinctive Image Features from Scale-Invariant Keypoints; IJCV"), [Dalal and Triggs 2005](<https://doi.org/10.1109/CVPR.2005.177> "Histograms of Oriented Gradients for Human Detection; CVPR").

### Signal and time-series transforms

- **Time-series statistics** - tsfresh combines a large feature bank with relevance filtering; catch22 provides 22 selected time-series characteristics. [Christ et al. 2018](<https://doi.org/10.1016/j.neucom.2018.03.067> "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package); Neurocomputing"), [Lubba et al. 2019](<https://doi.org/10.1007/s10618-019-00647-x> "catch22: CAnonical Time-series CHaracteristics: Selected through highly comparative time-series analysis; Data Min Knowl Disc"); [code](<https://github.com/blue-yonder/tsfresh>), [code](<https://github.com/DynamicsAndNeuralSystems/catch22>).
- **Wavelet scattering** - Cascades of wavelet transforms, nonlinearities, and averaging. Kymatio implements scattering for 1D, 2D, and 3D inputs. [Andreux et al. 2020](<https://www.jmlr.org/papers/v21/19-047.html> "Kymatio: Scattering Transforms in Python; JMLR"); [code](<https://github.com/kymatio/kymatio>).
- **Convolutional transforms** - ROCKET, MiniROCKET, and MultiROCKET produce convolution-response statistics for linear classification; extraction can expand dimensionality. [Dempster et al. 2020](<https://arxiv.org/abs/1910.13051> "ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels; Data Mining and Knowledge Discovery"), [Dempster et al. 2021](<https://arxiv.org/abs/2012.08791> "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification; KDD"), [Tan et al. 2022](<https://arxiv.org/abs/2102.00457> "MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification; Data Mining and Knowledge Discovery"); [code](<https://github.com/angus924/minirocket>).

## Tabular feature construction

- **OpenFE** - Expands tabular features with predefined operators, scores their incremental predictive value using feature boosting, and prunes candidates in two stages. Generation and supervised selection are coupled. [Zhang et al. 2023](<https://proceedings.mlr.press/v202/zhang23ay.html> "OpenFE: Automated Feature Generation with Expert-level Performance; ICML").
- **TabPrep** - Jointly tunes feature generators and downstream models, unlike OpenFE's separate feature selection. With four model families and post-hoc ensembling, improves the best score on 19 of 51 TabArena datasets against newer model baselines. [Tschalzev et al. 2026](<https://arxiv.org/abs/2606.02384v1> "TabPrep: Closing the Feature Engineering Gap in Tabular Benchmarks; NeurIPS").

## Linear and statistical representations

### Component analysis and factorization

- **PCA (principal component analysis)** - Orthogonal projection maximizing retained variance of centered data. [Pearson 1901](<https://zenodo.org/records/1430636> "LIII. On lines and planes of closest fit to systems of points in space; Philosophical Magazine"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>).
    - **Dual PCA** - Computes PCA through the sample Gram matrix. [Ghojogh and Crowley 2019](<https://arxiv.org/abs/1906.03148> "Unsupervised and Supervised Principal Component Analysis: Tutorial; arXiv").
    - **Incremental PCA** - Batchwise PCA updates. [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html>), [code](<https://cran.r-project.org/package=onlinePCA>).
- **Truncated SVD / latent semantic analysis** - Low-rank approximation without PCA's centering step, allowing sparse document-term inputs. [Deerwester et al. 1990](<https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9> "Indexing by latent semantic analysis; J. Am. Soc. Inf. Sci."); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>).
- **NMF (non-negative matrix factorization)** - Additive factorization of non-negative data. [Lee and Seung 1999 🔒](<https://doi.org/10.1038/44565> "Learning the parts of objects by non-negative matrix factorization; Nature"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html>).
- **Factor analysis** - A linear latent-factor model with feature-specific Gaussian noise. [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html>).
- **ICA (independent component analysis)** - Linear separation of statistically independent sources. [Hyvärinen and Oja 2000](<https://doi.org/10.1016/S0893-6080(00)00026-5> "Independent component analysis: algorithms and applications; Neural Networks"), [Hyvärinen et al. 2001](<https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf> "Independent Component Analysis; book"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html>).
- **Topic LDA (latent Dirichlet allocation)** - Document representations as topic mixtures. [Blei et al. 2003](<https://jmlr.org/papers/v3/blei03a.html> "Latent Dirichlet Allocation; JMLR"), [Hoffman et al. 2010](<https://www.di.ens.fr/~fbach/mdhnips2010.pdf> "Online Learning for Latent Dirichlet Allocation; NeurIPS").
- **Robust PCA / principal component pursuit** - Separates low-rank structure from sparse corruption (`X = L + S`), with recovery guarantees under rank, incoherence, and corruption assumptions. [Candes et al. 2011](<https://arxiv.org/abs/0912.3599> "Robust Principal Component Analysis?; JACM").

### Sparse representations

- **Sparse PCA** - Sparse component loadings, selecting which input variables contribute to each component. [Zou et al. 2006](<https://doi.org/10.1198/106186006X113430> "Sparse Principal Component Analysis; JCGS"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html>).
- **Dictionary learning / sparse coding** - Sparse sample codes over learned, possibly overcomplete dictionaries; sparsity applies to the codes rather than PCA loadings. [Mairal et al. 2010](<https://jmlr.org/papers/v11/mairal10a.html> "Online Learning for Matrix Factorization and Sparse Coding; JMLR"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html>).

### Supervised and multiview projections

- **Fisher LDA (linear discriminant analysis)** - Supervised projection maximizing between-class relative to within-class scatter. [Fisher 1936](<https://doi.org/10.1111/j.1469-1809.1936.tb02137.x> "The Use of Multiple Measurements in Taxonomic Problems; Annals of Eugenics"), [Rao 1948](<https://www.jstor.org/stable/2983775> "The Utilization of Multiple Measurements in Problems of Biological Classification; JRSS B"), [Martinez and Kak 2001](<https://doi.org/10.1109/34.908974> "PCA versus LDA; IEEE Trans. Pattern Anal. Machine Intell."); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html>).
- **Supervised PCA** - Response-based predictor screening followed by PCA. [Bair et al. 2006](<https://web.stanford.edu/~hastie/Papers/spca_JASA.pdf> "Prediction by Supervised Principal Components; JASA").
- **CCA / PLS** - Canonical correlation analysis maximizes correlation between projected views; partial least squares extracts cross-covariance-based components, including for regression. [docs](<https://scikit-learn.org/stable/modules/cross_decomposition.html>).

## Kernel methods and random projections

- **Kernel PCA** - Component analysis in an implicit kernel feature space. [Schölkopf et al. 1998](<https://doi.org/10.1162/089976698300017467> "Nonlinear Component Analysis as a Kernel Eigenvalue Problem; Neural Comput."); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html>).
    - **Kernel PCA extensions** - Kernel PCA for novelty detection and robust kernel PCA. [Hoffmann 2007](<https://www.heikohoffmann.de/documents/hoffmann_kpca_preprint.pdf> "Kernel PCA for Novelty Detection; Pattern Recognition"), [Nguyen and De la Torre 2008](<https://papers.nips.cc/paper/3566-robust-kernel-principal-component-analysis> "Robust Kernel Principal Component Analysis; Advances in Neural Information Processing Systems").
- **Kernel Fisher discriminant analysis** - Supervised class separation in a kernel feature space. [Mika et al. 1999](<https://doi.org/10.1109/NNSP.1999.788121> "Fisher discriminant analysis with kernels; NNSP").
- **Nyström approximation** - Sampled data landmarks approximate the kernel matrix; an alternative to Fourier sampling. [Williams and Seeger 2000](<https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html> "Using the Nyström Method to Speed Up Kernel Machines; NeurIPS"); [docs](<https://scikit-learn.org/stable/modules/kernel_approximation.html>).
- **Random projections** - Gaussian or sparse random maps approximately preserve Euclidean distances under Johnson-Lindenstrauss conditions. [Achlioptas 2003](<https://doi.org/10.1016/S0022-0000(03)00025-4> "Database-friendly random projections: Johnson-Lindenstrauss with binary coins; JCSS"); [docs](<https://scikit-learn.org/stable/modules/random_projection.html>).
- **Random Fourier features** - Sampled Fourier maps approximate shift-invariant kernels. Unlike distance-preserving random projections, these construct nonlinear features and may expand dimension. [Rahimi and Recht 2007](<https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines> "Random Features for Large-Scale Kernel Machines; NeurIPS").

## Manifold learning and visualization

### Distance and spectral methods

- **MDS (multidimensional scaling)** - Fits coordinates to dissimilarities or their ordering. [Kruskal 1964](<https://doi.org/10.1007/BF02289565> "Multidimensional Scaling by Optimizing Goodness of Fit to a Nonmetric Hypothesis; Psychometrika"), [Little et al. 2018](<https://arxiv.org/abs/1812.11954> "Exact Cluster Recovery via Classical Multidimensional Scaling; arXiv"), [Little et al. 2022](<https://doi.org/10.1093/imaiai/iaac004> "An analysis of classical multidimensional scaling with applications to clustering; Information and Inference: A Journal of the IMA"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html>).
- **Sammon mapping** - Weighted distance reconstruction emphasizing smaller input distances. [Sammon 1969](<https://doi.org/10.1109/T-C.1969.222678> "A Nonlinear Mapping for Data Structure Analysis; IEEE Trans. Comput.").
- **SOM (self-organizing maps)** - Organizes adaptive units into a low-dimensional array whose responses reflect input topology. Kohonen also examines conditions where ordering fails. [Kohonen 1982 🔒](<https://doi.org/10.1007/BF00337288> "Self-organized formation of topologically correct feature maps; Biol. Cybern.").
- **Isomap** - MDS on graph-approximated geodesic distances. [Tenenbaum et al. 2000](<https://doi.org/10.1126/science.290.5500.2319> "A Global Geometric Framework for Nonlinear Dimensionality Reduction; Science"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html>).
- **LLE (locally linear embedding)** - Fits coordinates using fixed local linear reconstruction weights. [Roweis and Saul 2000](<https://doi.org/10.1126/science.290.5500.2323> "Nonlinear Dimensionality Reduction by Locally Linear Embedding; Science"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html>).
    - **HLLE (Hessian eigenmapping)** - Manifold coordinates from local Hessian estimates. Supported by scikit-learn's LLE interface. [Donoho and Grimes 2003](<https://doi.org/10.1073/pnas.1031596100> "Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data; PNAS").
- **Laplacian eigenmaps / spectral embedding** - Graph-local similarity represented by Laplacian eigenvectors. [Belkin and Niyogi 2003](<https://doi.org/10.1162/089976603321780317> "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation; Neural Comput.").
- **Maximum variance unfolding / semidefinite embedding** - Maximizes spread while preserving local distances through semidefinite optimization. [Weinberger et al. 2005](<https://proceedings.mlr.press/r5/weinberger05a.html> "Nonlinear Dimensionality Reduction by Semidefinite Programming and Kernel Matrix Factorization; AISTATS").
- **Diffusion maps** - Spectral coordinates reflecting random-walk connectivity across scales. [Coifman and Lafon 2006](<https://doi.org/10.1016/j.acha.2006.04.006> "Diffusion maps; ACHA").
    - **PHATE** - Embeds distances between transformed diffusion probabilities to visualize continuous and branching structure. [Moon et al. 2019 🔒](<https://doi.org/10.1038/s41587-019-0336-3> "Visualizing structure and transitions in high-dimensional biological data; Nat Biotechnol"); [code](<https://github.com/KrishnaswamyLab/PHATE>).

### Neighborhood objectives and learned mappings

- **t-SNE** - Local neighborhood probability matching. [van der Maaten and Hinton 2008](<https://jmlr.org/papers/v9/vandermaaten08a.html> "Visualizing Data using t-SNE; JMLR"), [van der Maaten 2014](<https://jmlr.org/papers/v15/vandermaaten14a.html> "Accelerating t-SNE using Tree-Based Algorithms; Journal of Machine Learning Research"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>), [docs](<https://opentsne.readthedocs.io/>).
    - **Extensions** - Tree-SNE for hierarchical clustering; LEt-SNE for hyperspectral imagery. [Robinson and Pierce-Hoffman 2020](<https://arxiv.org/abs/2002.05687> "Tree-SNE: Hierarchical Clustering and Visualization Using t-SNE; arXiv"), [Shukla et al. 2019](<https://arxiv.org/abs/1910.08790> "LEt-SNE: A Hybrid Approach To Data Embedding and Visualization of Hyperspectral Imagery; arXiv"); [code](<https://github.com/isaacrob/treesne>).
- **LargeVis** - Approximate neighborhood-graph construction and probabilistic layout for large collections. [Tang et al. 2016](<https://arxiv.org/abs/1602.00370> "Visualizing Large-scale and High-dimensional Data; WWW").
- **UMAP** - Embeds a fuzzy neighborhood graph. [McInnes et al. 2018](<https://arxiv.org/abs/1802.03426> "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction; arXiv"); [code](<https://github.com/lmcinnes/umap>), [docs](<https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.manifold.UMAP/>).
    - **Parametric UMAP** - Trains a neural encoder with the UMAP objective. [Sainburg et al. 2021](<https://arxiv.org/abs/2009.12981> "Parametric UMAP embeddings for representation and semi-supervised learning; Neural Comput."); [docs](<https://umap-learn.readthedocs.io/en/latest/parametric_umap.html>).
    - **densMAP** - Adds relative local-density preservation to UMAP. [Narayan et al. 2021 🔒](<https://doi.org/10.1038/s41587-020-00801-7> "Assessing single-cell transcriptomic variability through density-preserving data visualization; Nat Biotechnol"); [docs](<https://umap-learn.readthedocs.io/en/latest/densmap_demo.html>).
    - **MAPLE** - Adds self-supervised manifold modeling to UMAP using maximum manifold capacity representations. [Huang et al. 2026](<https://arxiv.org/abs/2601.20173> "MAPLE: Self-Supervised Learning-Enhanced Nonlinear Dimensionality Reduction for Visual Analysis; TVCG"), [Huang et al. 2026](<https://doi.org/10.1109/TVCG.2026.3694000> "MAPLE: Self-Supervised Learning-Enhanced Nonlinear Dimensionality Reduction for Visual Analysis; IEEE Trans. Visual. Comput. Graphics").
- **TriMap** - Relative similarity preservation through triplets. [Amid and Warmuth 2019](<https://arxiv.org/abs/1910.00204> "TriMap: Large-scale Dimensionality Reduction Using Triplets; arXiv"); [code](<https://github.com/eamid/trimap>).
- **PaCMAP** - Combines neighbor, mid-near, and distant pairs to preserve structure at different scales. [Wang et al. 2021](<https://arxiv.org/abs/2012.04456> "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization; JMLR"); [code](<https://github.com/YingfanWang/PaCMAP>).
    - **LocalMAP** - Adjusts the neighborhood graph during optimization to improve local cluster separation. Implemented in the PaCMAP repository. [Wang et al. 2025](<https://arxiv.org/abs/2412.15426> "Dimension Reduction with Locally Adjusted Graphs; AAAI").
- **ParamRepulsor** - Hard-negative mining and stronger repulsion address loss of local detail in parametric embeddings. [Huang et al. 2024](<https://arxiv.org/abs/2411.15894> "Navigating the Effect of Parametrization for Dimensionality Reduction; NeurIPS"); [code](<https://github.com/hyhuang00/ParamRepulsor>).
- **New observations** - Ordinary UMAP and openTSNE place new points relative to a fitted embedding. Parametric UMAP instead evaluates a learned encoder; new-point support alone does not imply a neural mapping. [docs](<https://umap-learn.readthedocs.io/en/latest/transform.html>), [docs](<https://opentsne.readthedocs.io/en/stable/api/index.html#openTSNE.TSNEEmbedding.transform>).

## Competitive and modular representation learning

- **Competitive learning** - Rumelhart and Zipser study how competition among units discovers feature detectors without target labels, including multilayer representations of nonlinearly separable patterns. [Rumelhart and Zipser 1985](<https://doi.org/10.1207/s15516709cog0901_5> "Feature Discovery by Competitive Learning*; Cognitive Science").
- **Modular autoassociative learning** - Couples autoassociative modules into hierarchies to learn internal representations. An early approach to organizing learning in networks with multiple hidden levels. [Ballard 1987](<https://www.aaai.org/Papers/AAAI/1987/AAAI87-050.pdf> "Modular Learning in Neural Networks; AAAI").

## Learned representations

### Reconstruction and masking

- **Autoencoders** - Encoder-decoder learning through reconstruction. [Hinton and Salakhutdinov 2006](<https://www.cs.toronto.edu/~hinton/absps/science.pdf> "Reducing the Dimensionality of Data with Neural Networks; Science").
    - **Denoising** - Learns representations by reconstructing clean inputs from corrupted versions. Stacked denoising modules can initialize deeper networks. [Vincent et al. 2008](<https://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf> "Extracting and composing robust features with denoising autoencoders; ICML"), [Vincent et al. 2010](<https://jmlr.org/papers/v11/vincent10a.html> "Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion; JMLR").
    - **Contractive autoencoders** - Adds a penalty on the encoder's input Jacobian to reconstruction loss, encouraging local insensitivity of the representation to input changes. [Rifai et al. 2011](<https://www.iro.umontreal.ca/~lisa/pointeurs/ICML2011_explicit_invariance.pdf> "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction; ICML").
    - **Variational autoencoders** - See [Generative Models](https://mlpapers.org/generative-models/#latent-variable-models) for the latent distribution, approximate posterior and variational objective.
    - **Masked** - Encodes visible image patches and reconstructs missing pixels with a lightweight decoder. Representation learning operates on the visible subset, separating encoder cost from full-image reconstruction. [He et al. 2022](<https://arxiv.org/abs/2111.06377> "Masked Autoencoders Are Scalable Vision Learners; CVPR"); [code](<https://github.com/facebookresearch/mae>).
- **Autoencoder objectives** - Learns an encoder and reconstruction decoder under restrictions that prevent unrestricted copying. Covers bottlenecks, regularization, and links to latent-variable models. [Goodfellow et al. 2016](<https://www.deeplearningbook.org/contents/autoencoders.html> "Deep Learning; book").

### Contrastive learning

- **Contrastive Predictive Coding** - Predicts future latent representations from autoregressive context using a contrastive loss and negative samples; learns sequential features without reconstructing raw observations. [van den Oord et al. 2018](<https://arxiv.org/abs/1807.03748> "Representation Learning with Contrastive Predictive Coding; arXiv").
- **SimCLR** - Contrasts augmented image views through a nonlinear projection head. Augmentation composition and batch size affect learning. ImageNet linear evaluation favors encoder features before the head over its output. [Chen et al. 2020](<https://arxiv.org/abs/2002.05709> "A Simple Framework for Contrastive Learning of Visual Representations; ICML"); [code](<https://github.com/google-research/simclr>).
- **Alignment and uniformity** - Relates contrastive learning to positive-pair alignment and uniform spread on the unit sphere. The analysis is asymptotic; normalization alone does not guarantee uniform learned features. [Wang and Isola 2020](<https://proceedings.mlr.press/v119/wang20k.html> "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere; ICML").

### Self-distillation and redundancy reduction

- **BYOL** - An online network predicts a moving-average target network's representation of another augmented view, without explicit negative pairs. [Grill et al. 2020](<https://arxiv.org/abs/2006.07733> "Bootstrap your own latent: A new approach to self-supervised Learning; NeurIPS").
- **DINO** - Uses self-distillation without labels, a momentum teacher and multiple crops. Studies semantic structure in Transformer features through nearest-neighbor and linear classification. [Caron et al. 2021](<https://arxiv.org/abs/2104.14294> "Emerging Properties in Self-Supervised Vision Transformers; ICCV").
    - **DINOv2** - Combines curated image data, scaled self-supervised training and distillation to produce reusable image-level and patch-level features. [Oquab et al. 2024](<https://arxiv.org/abs/2304.07193> "DINOv2: Learning Robust Visual Features without Supervision; TMLR"); [code](<https://github.com/facebookresearch/dinov2>).
    - **DINOv3** - Uses Gram anchoring to address deterioration of dense features during extended self-supervised training. Studies scaling and post-training adaptation of visual representations. [Siméoni et al. 2026](<https://arxiv.org/abs/2508.10104> "DINOv3; NeurIPS"); [code](<https://github.com/facebookresearch/dinov3>).
- **Barlow Twins** - Drives cross-correlation between two augmented views toward identity, combining agreement with redundancy reduction. This objective also underlies [TLDR](#representation-compression). [Zbontar et al. 2021](<https://arxiv.org/abs/2103.03230> "Barlow Twins: Self-Supervised Learning via Redundancy Reduction; ICML").
- **VICReg** - Separates view agreement, per-feature variance control, and covariance regularization. Unlike Barlow Twins' cross-correlation objective, it explicitly constrains each branch's variance. [Bardes et al. 2022](<https://arxiv.org/abs/2105.04906> "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning; ICLR"); [code](<https://github.com/facebookresearch/vicreg>).

### Prediction in representation space

- **Self-predictive representations (SPR)** - Predicts future encodings from features and actions as an auxiliary loss for Rainbow. Improves Atari 100k performance over a matched augmented Rainbow baseline without using the learned dynamics for planning. [Schwarzer et al. 2021](<https://arxiv.org/abs/2007.05929v4> "Data-Efficient Reinforcement Learning with Self-Predictive Representations; ICLR"); [code](<https://github.com/mila-iqia/spr>).
- **I-JEPA** - Predicts representations of masked image regions from visible context, rather than reconstructing pixels as in MAE. [Assran et al. 2023](<https://arxiv.org/abs/2301.08243> "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture; CVPR").
- **JEPA learning bias** - In deep linear models, JEPA's feature-learning order depends on regression coefficients and input-target covariance; masked reconstruction is led by covariance. Theory assumes diagonal covariances, small aligned initialization and stop-gradient targets. [Littwin et al. 2024](<https://proceedings.neurips.cc/paper_files/paper/2024/hash/a600f0f740605205133553cb74a1c107-Abstract-Conference.html> "How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks; NeurIPS").
- **LeJEPA** - Combines a joint-embedding predictive objective with sketched isotropic Gaussian regularization of embeddings. [Balestriero and LeCun 2025](<https://arxiv.org/abs/2511.08544> "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics; arXiv").
    - **LeVJEPA** - Extends LeJEPA to video using one encoder and 95% patch-token dropping. At matched pretraining FLOPs on 20% of K710, ViT-B improves ImageNet probing over V-JEPA 2 (61.0% vs 51.6%) but trails on Something-Something-v2 (40.4% vs 42.5%). [Kuhn et al. 2026](<https://arxiv.org/abs/2608.27395v1> "LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics; arXiv"); [code](<https://github.com/MLO-lab/LeVJEPA>).
- **JEPA predictive closure** - In CIFAR-10 sequences with three-dimensional SIGReg latents and no stop-gradient, an unpredictable cue becomes more decodable when it helps predict other features. Altering predictive links shifts which feature sets are represented. [Levy et al. 2026](<https://arxiv.org/abs/2610.27364v1> "Predictive Closure: A Set-Level Characterization of JEPA’s Feature Selection; arXiv"); [code](<https://github.com/Mosh0110/predictive-closure>).
- **V-JEPA 2.1** - Dense prediction over visible and masked tokens, plus intermediate-layer self-supervision, extends JEPA features to spatially structured image/video representations. [Mur-Labadia et al. 2026](<https://arxiv.org/abs/2603.14482> "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning; ECCV"); [code](<https://github.com/facebookresearch/vjepa2>).
- **Spectral Ghost** - Develops a spectral account of representation sufficiency connecting component analysis and self-supervised objectives. [Dai et al. 2026](<https://arxiv.org/abs/2601.20154> "Spectral Ghost in Representation Learning: from Component Analysis to Self-Supervised Learning; arXiv").
- **Cross-modal agreement** - See [Interpretability](https://mlpapers.org/interpretability/#comparing-representations) for the Platonic representation hypothesis, similarity calibration and JEPA's scene-pairing results.

## Identifiability and disentanglement

- **Limits of unsupervised disentanglement** - Locatello et al. show that recovering independent factors requires inductive assumptions. Across seven datasets, stronger disentanglement does not consistently reduce downstream sample needs; factor recovery is a distinct goal. [Locatello et al. 2019](<https://arxiv.org/abs/1811.12359> "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations; ICML").
- **Identifiable VAE / nonlinear ICA** - Conditions a factorized latent prior on an observed auxiliary variable. Identifiability holds up to specified transformations under variation and regularity conditions; an ordinary VAE with an unconditional prior lacks this guarantee. [Khemakhem et al. 2020](<https://arxiv.org/abs/1907.04809> "Variational Autoencoders and Nonlinear ICA: A Unifying Framework; AISTATS").

## Extraction by modality

### Images and video

- **Visual representations** - See [self-distillation](#self-distillation-and-redundancy-reduction) for DINO and [predictive representations](#prediction-in-representation-space) for JEPA. [Transfer Learning](https://mlpapers.org/transfer-learning/#frozen-features-and-fine-tuning) covers adaptation.

### Text and retrieval

<a id="sentence-and-document-vectors"></a>
<a id="document-context-and-chunk-pooling"></a>
<a id="sparse-and-multivector-scoring"></a>

- **Text embeddings and retrieval** - See [NLP](https://mlpapers.org/nlp/#sentence-representations-and-retrieval) for sentence vectors, token scoring and document-context-aware chunk embeddings.

### Audio and time series

#### Audio representations

- **wav2vec 2.0** - Masks latent speech representations and predicts jointly learned quantized targets with a contrastive loss. Unlabeled audio pretraining precedes supervised speech-recognition fine-tuning. [Baevski et al. 2020](<https://arxiv.org/abs/2006.11477> "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations; NeurIPS").
- **BEATs** - Iterates acoustic tokenization and masked audio-token prediction to learn general audio representations. [Chen et al. 2023](<https://proceedings.mlr.press/v202/chen23ag.html> "BEATs: Audio Pre-Training with Acoustic Tokenizers; ICML"); [code](<https://github.com/microsoft/unilm/tree/master/beats>).

#### Time-series representations

- **TS2Vec** - Contrastive learning of contextual timestamp features, pooled for subsequences or whole series. [Yue et al. 2022](<https://arxiv.org/abs/2106.10466> "TS2Vec: Towards Universal Representation of Time Series; AAAI"); [code](<https://github.com/zhihanyue/ts2vec>).
- **MOMENT** - Masked time-series pretraining supports classification, forecasting, imputation, and anomaly detection. [Goswami et al. 2024](<https://arxiv.org/abs/2402.03885> "MOMENT: A Family of Open Time-series Foundation Models; ICML"); [code](<https://github.com/moment-timeseries-foundation-model/moment>).
- **MantisV2** - Uses synthetic pretraining, layer selection and token aggregation for frozen time-series features. UCR ablations favor intermediate layers after more updates, but the final layer with 100K samples and 100 epochs; fine-tuning favors all layers. [Feofanov et al. 2026](<https://arxiv.org/abs/2602.17868> "MantisV2: Closing the Zero-Shot Gap in Time Series Classification with Synthetic Data and Test-Time Strategies; arXiv"); [code](<https://github.com/vfeofanov/mantis>).

### Cross-modal representations

- **Joint representations** - See [Multi-View Learning](https://mlpapers.org/multiview-learning/#cross-modal-alignment-and-fusion) for CLIP, CLAP, SigLIP 2 and multimodal retrieval.

## Representation compression

- **Matryoshka representation learning** - Trains nested vector prefixes at multiple dimensions. Compression is built into representation training; arbitrary truncation of ordinary embeddings is not equivalent. [Kusupati et al. 2022](<https://arxiv.org/abs/2205.13147> "Matryoshka Representation Learning; NeurIPS"); [code](<https://github.com/RAIVNLab/MRL>).
- **TLDR** - Compresses features using neighbor pairs and the Barlow Twins objective. On ROxford retrieval, the reported 128-dimensional codes gain 4.1 mAP points over whitened PCA with the tested GeM features and fitting data. [Kalantidis et al. 2022](<https://arxiv.org/abs/2110.09455> "TLDR: Twin Learning for Dimensionality Reduction; TMLR"); [code](<https://github.com/naver/tldr>).
- **Retrieval dimension and margin** - For unit-vector retrieval supporting every k-subset of n documents, optimal-order margin requires dimension of order k log(n/k). This worst-case relevance family does not determine the dimension needed for a particular corpus. [Bangachev et al. 2026](<https://arxiv.org/abs/2605.23556v1> "Is Dimensionality a Barrier for Retrieval Models?; NeurIPS").
- **Multivector expressivity** - Proves a worst-case representation-size separation between Chamfer similarity and single-vector inner products; this is not an empirical retrieval ranking. [Jayaram 2026](<https://arxiv.org/abs/2606.23475> "Multi-Vector Embeddings are Provably More Expressive than Single Vector Embeddings; arXiv").

## Evaluation and comparative evidence

### Benchmarks and geometry

- **Neighborhood preservation** - Trustworthiness penalizes introduced neighbors; continuity penalizes lost neighbors. These measure geometry rather than held-out task performance. [Venna and Kaski 2006](<https://doi.org/10.1016/j.neunet.2006.05.014> "Local multidimensional scaling; Neural Networks"); [docs](<https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html>).
- **Intrinsic dimension** - Global and local estimators characterize data geometry; their estimates are not task-optimized embedding dimensions. [Bac et al. 2021](<https://arxiv.org/abs/2109.02596> "Scikit-dimension: a Python package for intrinsic dimension estimation; Entropy"); [docs](<https://scikit-dimension.readthedocs.io/>).
- **Local versus global structure** - On tested 2D layouts, t-SNE/art-SNE preserve neighborhoods better, while PCA/TriMap/PaCMAP/ForceAtlas2 preserve distance ranks better on most datasets. These visualization results do not rank general predictive representations. [Huang et al. 2022](<https://pmc.ncbi.nlm.nih.gov/articles/PMC9296444/> "Towards a comprehensive evaluation of dimension reduction methods for transcriptomic data visualization; Commun. Biol.").
- **MTEB / MMTEB** - Text-embedding evaluation across tasks, domains, and languages. [Muennighoff et al. 2023](<https://arxiv.org/abs/2210.07316> "MTEB: Massive Text Embedding Benchmark; EACL"), [Enevoldsen et al. 2025](<https://arxiv.org/abs/2502.13595> "MMTEB: Massive Multilingual Text Embedding Benchmark; ICLR"); [code](<https://github.com/embeddings-benchmark/mteb>).
- **Prediction targets for control** - Latent prediction is more robust than observation prediction to added Gaussian distractors in four MuJoCo tasks. Observation prediction instead improves aggregate learning on 20 sparse-reward, partially observed MiniGrid tasks. [Ni et al. 2024](<https://proceedings.iclr.cc/paper_files/paper/2024/file/666c1861d709bd84e20b6e0e02a2c223-Paper-Conference.pdf> "Bridging State and History Representations: Understanding Self-Predictive RL; ICLR").

<a id="extraction-choices"></a>

## Software

- **Python (reduction and dimension estimation)** - scikit-learn, umap-learn, openTSNE, PaCMAP/LocalMAP, PHATE, scikit-dimension. [docs](<https://scikit-learn.org/stable/>), [docs](<https://umap-learn.readthedocs.io/>), [docs](<https://opentsne.readthedocs.io/>), [code](<https://github.com/YingfanWang/PaCMAP>), [code](<https://github.com/KrishnaswamyLab/PHATE>), [docs](<https://scikit-dimension.readthedocs.io/>).
- **Python (extraction)** - Sentence Transformers, timm, Kymatio, tsfresh; evaluation: MTEB. [docs](<https://www.sbert.net/>), [docs](<https://huggingface.co/docs/timm/feature_extraction>), [docs](<https://www.kymat.io/>), [docs](<https://tsfresh.readthedocs.io/>), [code](<https://github.com/embeddings-benchmark/mteb>).
- **R** - dimRed, dyndimred, intrinsicDimension, Rdimtools; discriminant analysis: MASS, sda. [You 2022](<https://arxiv.org/abs/2005.11107> "Rdimtools: An R package for Dimension Reduction and Intrinsic Dimension Estimation; Software Impacts"); [code](<https://cran.r-project.org/package=dimRed>), [code](<https://cran.r-project.org/package=dyndimred>), [code](<https://cran.r-project.org/package=intrinsicDimension>), [code](<https://cran.r-project.org/package=Rdimtools>), [code](<https://cran.r-project.org/package=MASS>), [code](<https://cran.r-project.org/package=sda>).
- **JavaScript (extraction)** - Transformers.js runs compatible pretrained models through ONNX Runtime with WASM/WebGPU. [docs](<https://huggingface.co/docs/transformers.js/index>).
- **JavaScript (visualization)** - tsne, umap-js, dimred; interactive tool: StatSim Vis. [docs](<https://www.npmjs.com/package/tsne>), [docs](<https://www.npmjs.com/package/umap-js>), [docs](<https://www.npmjs.com/package/dimred>), [docs](<https://statsim.com/vis/>).
- **C++** - Tapkee. [code](<https://github.com/lisitsyn/tapkee>).

## Connections

- **Prediction and control** - [MuZero](https://mlpapers.org/reinforcement-learning/#learned-dynamics-and-planning) predicts rewards, values and search policies. EfficientZero adds future-latent matching. [V-JEPA 2-AC](https://mlpapers.org/model-predictive-control/#latent-planning-and-comparative-evidence) plans with image-goal distances after self-supervised training.
- **Sparse autoencoders for interpretability** - Overcomplete sparse features from internal model activations, evaluated for reconstruction and interpretability rather than compactness alone. [Gao et al. 2024](<https://arxiv.org/abs/2406.04093> "Scaling and evaluating sparse autoencoders; arXiv").

## Related topics

- [Reinforcement learning](https://mlpapers.org/reinforcement-learning/)
- [Model predictive control](https://mlpapers.org/model-predictive-control/)
- [Feature selection](https://mlpapers.org/feature-selection/)
- [Neural networks](https://mlpapers.org/neural-nets/)
- [Multiview learning](https://mlpapers.org/multiview-learning/)
- [Interpretability](https://mlpapers.org/interpretability/)
