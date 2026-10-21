# Feature Extraction and Representation Learning

## Surveys

- [Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538) (2012; TPAMI 2013) — Bengio, Courville, Vincent. Probabilistic models, autoencoders, manifold learning.
- [A Survey of Dimensionality Reduction Techniques](https://arxiv.org/abs/1403.2877) (2014) — Sorzano, Vargas, Pascual-Montano.
- [Feature Selection and Feature Extraction in Pattern Analysis](https://arxiv.org/abs/1905.02845) (2019) — Ghojogh et al.

## Fixed and domain-informed features

### Text and image descriptors

- **Text vectorization:** sparse word/character n-grams, TF–IDF, and feature hashing. [scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction).
- **Image descriptors:** [SIFT](https://doi.org/10.1023/B:VISI.0000029664.99615.94) (2004) describes local keypoints; [HOG](https://doi.org/10.1109/CVPR.2005.177) (2005) summarizes local gradient orientations.

### Signal and time-series transforms

- **Wavelet scattering:** cascades of wavelet transforms, nonlinearities, and averaging. [Kymatio](https://www.jmlr.org/papers/v21/19-047.html) (2020) implements scattering for 1D, 2D, and 3D inputs. [Code](https://github.com/kymatio/kymatio).
- **Time-series statistics:** [tsfresh](https://doi.org/10.1016/j.neucom.2018.03.067) (2018; [code](https://github.com/blue-yonder/tsfresh)) combines a large feature bank with relevance filtering; [catch22](https://doi.org/10.1007/s10618-019-00647-x) (2019; [code](https://github.com/DynamicsAndNeuralSystems/catch22)) provides 22 selected time-series characteristics.
- **Convolutional transforms:** [ROCKET](https://arxiv.org/abs/1910.13051) (2019/2020), [MiniROCKET](https://arxiv.org/abs/2012.08791) (2020/2021; [code](https://github.com/angus924/minirocket)), and [MultiROCKET](https://arxiv.org/abs/2102.00457) (2021/2022) produce convolution-response statistics for linear classification; extraction can expand dimensionality.

## Linear and statistical representations

### Component analysis and factorization

- **PCA — principal component analysis:** orthogonal projection maximizing retained variance of centered data. [Pearson](https://zenodo.org/records/1430636) (1901); [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
    - **Dual PCA:** computes PCA through the sample Gram matrix. [Derivation](https://arxiv.org/abs/1906.03148) (2019).
    - **Incremental PCA:** batchwise PCA updates. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html), [onlinePCA for R](https://cran.r-project.org/package=onlinePCA).
- **Truncated SVD / latent semantic analysis:** low-rank approximation without PCA's centering step, allowing sparse document-term inputs. [LSA](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9) (1990); [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
- **ICA — independent component analysis:** linear separation of statistically independent sources. [Review](https://doi.org/10.1016/S0893-6080(00)00026-5) (2000), [book](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf) (2001); [FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html).
- **Factor analysis:** a linear latent-factor model with feature-specific Gaussian noise. [Model and implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html).
- **NMF — non-negative matrix factorization:** additive factorization of non-negative data. [Lee and Seung](https://doi.org/10.1038/44565) (1999); [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).
- **Robust PCA / principal component pursuit:** separates low-rank structure from sparse corruption (`X = L + S`), with recovery guarantees under rank, incoherence, and corruption assumptions. [Candès et al.](https://arxiv.org/abs/0912.3599) (2009; JACM 2011).
- **Topic LDA — latent Dirichlet allocation:** document representations as topic mixtures. [Blei, Ng, Jordan](https://jmlr.org/papers/v3/blei03a.html) (2003); [online variational learning](https://www.di.ens.fr/~fbach/mdhnips2010.pdf) (2010).

### Sparse representations

- **Sparse PCA:** sparse component loadings, selecting which input variables contribute to each component. [Implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html).
- **Dictionary learning / sparse coding:** sparse sample codes over learned, possibly overcomplete dictionaries; sparsity applies to the codes rather than PCA loadings. [Mairal et al.](https://jmlr.org/papers/v11/mairal10a.html) (2010); [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html).

### Supervised and multiview projections

- **Fisher LDA — linear discriminant analysis:** supervised projection maximizing between-class relative to within-class scatter. [Fisher](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) (1936), [Rao](https://www.jstor.org/stable/2983775) (1948), [PCA versus LDA](https://doi.org/10.1109/34.908974) (2001); [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).
- **Supervised PCA:** response-based predictor screening followed by PCA. [Bair et al.](https://web.stanford.edu/~hastie/Papers/spca_JASA.pdf) (2006).
- **CCA / PLS:** canonical correlation analysis maximizes correlation between projected views; partial least squares extracts cross-covariance-based components, including for regression. [Objectives and implementations](https://scikit-learn.org/stable/modules/cross_decomposition.html).

## Kernel methods and random projections

- **Random projections:** Gaussian or sparse random maps approximately preserve Euclidean distances under Johnson–Lindenstrauss conditions. [Theory and implementations](https://scikit-learn.org/stable/modules/random_projection.html).
- **Kernel PCA:** component analysis in an implicit kernel feature space. [Schölkopf, Smola, Müller](https://doi.org/10.1162/089976698300017467) (1998); [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html).
    - Extensions: [novelty detection](https://www.heikohoffmann.de/documents/hoffmann_kpca_preprint.pdf) (2007) — Hoffmann; [robust kernel PCA](https://papers.nips.cc/paper/3566-robust-kernel-principal-component-analysis) (2008) — Nguyen, De la Torre.
- **Kernel Fisher discriminant analysis:** supervised class separation in a kernel feature space. [Mika et al.](https://doi.org/10.1109/NNSP.1999.788121) (1999).
- **Random Fourier features:** sampled Fourier maps approximate shift-invariant kernels. Unlike distance-preserving random projections, these construct nonlinear features and may expand dimension. [Rahimi and Recht](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) (2007).
- **Nyström approximation:** sampled data landmarks approximate the kernel matrix; an alternative to Fourier sampling. [Williams and Seeger](https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html) (NIPS 2000); [Nystroem and RBFSampler](https://scikit-learn.org/stable/modules/kernel_approximation.html).

## Manifold learning and visualization

### Distance and spectral methods

- **MDS — multidimensional scaling:** fits coordinates to dissimilarities or their ordering. [Nonmetric MDS](https://doi.org/10.1007/BF02289565) (1964); [exact cluster recovery with classical MDS](https://arxiv.org/abs/1812.11954) (2018; [journal 2023](https://doi.org/10.1093/imaiai/iaac004)); [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html).
- **Sammon mapping:** weighted distance reconstruction emphasizing smaller input distances. [Sammon](https://doi.org/10.1109/T-C.1969.222678) (1969).
- **SOM — self-organizing maps:** updates grid-organized prototypes and their neighbors. [Kohonen](https://doi.org/10.1007/BF00337288) (1982).
- **Isomap:** MDS on graph-approximated geodesic distances. [Tenenbaum, de Silva, Langford](https://doi.org/10.1126/science.290.5500.2319) (2000); [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html).
- **LLE — locally linear embedding:** fits coordinates using fixed local linear reconstruction weights. [Roweis and Saul](https://doi.org/10.1126/science.290.5500.2323) (2000); [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html).
    - **HLLE — Hessian eigenmapping:** manifold coordinates from local Hessian estimates. [Donoho and Grimes](https://doi.org/10.1073/pnas.1031596100) (2003); supported by scikit-learn's LLE interface.
- **Laplacian eigenmaps / spectral embedding:** graph-local similarity represented by Laplacian eigenvectors. [Belkin and Niyogi](https://doi.org/10.1162/089976603321780317) (2003).
- **Maximum variance unfolding / semidefinite embedding:** maximizes spread while preserving local distances through semidefinite optimization. [Weinberger, Packer, Saul](https://proceedings.mlr.press/r5/weinberger05a.html) (AISTATS 2005).
- **Diffusion maps:** spectral coordinates reflecting random-walk connectivity across scales. [Coifman and Lafon](https://doi.org/10.1016/j.acha.2006.04.006) (2006).
    - **PHATE:** embeds distances between transformed diffusion probabilities to visualize continuous and branching structure. [Moon et al.](https://doi.org/10.1038/s41587-019-0336-3) (2019); [code](https://github.com/KrishnaswamyLab/PHATE).

### Neighborhood objectives and learned mappings

- **t-SNE:** local neighborhood probability matching. [van der Maaten and Hinton](https://jmlr.org/papers/v9/vandermaaten08a.html) (2008); [tree-based acceleration](https://jmlr.org/papers/v15/vandermaaten14a.html) (2014); [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [openTSNE](https://opentsne.readthedocs.io/).
    - Extensions: [Tree-SNE](https://arxiv.org/abs/2002.05687) (2020; [code](https://github.com/isaacrob/treesne)) for hierarchical clustering; [LEt-SNE](https://arxiv.org/abs/1910.08790) (2019; ICASSP 2020) for hyperspectral imagery.
- **LargeVis:** approximate neighborhood-graph construction and probabilistic layout for large collections. [Tang et al.](https://arxiv.org/abs/1602.00370) (2016).
- **UMAP:** embeds a fuzzy neighborhood graph. [McInnes, Healy, Melville](https://arxiv.org/abs/1802.03426) (2018); [code](https://github.com/lmcinnes/umap), [GPU implementation](https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.manifold.UMAP/).
    - **Parametric UMAP:** trains a neural encoder with the UMAP objective. [Paper](https://arxiv.org/abs/2009.12981) (2020/2021); [implementation](https://umap-learn.readthedocs.io/en/latest/parametric_umap.html).
    - **densMAP:** adds relative local-density preservation to UMAP. [Paper](https://doi.org/10.1038/s41587-020-00801-7) (2021); [implementation](https://umap-learn.readthedocs.io/en/latest/densmap_demo.html).
    - **MAPLE:** adds self-supervised manifold modeling to UMAP using maximum manifold capacity representations. [Paper](https://arxiv.org/abs/2601.20173) ([TVCG 2026](https://doi.org/10.1109/TVCG.2026.3694000)).
- **TriMap:** relative similarity preservation through triplets. [Paper](https://arxiv.org/abs/1910.00204) (2019); [code](https://github.com/eamid/trimap).
- **PaCMAP:** combines neighbor, mid-near, and distant pairs to preserve structure at different scales. [Paper](https://arxiv.org/abs/2012.04456) (2020; JMLR 2021); [code](https://github.com/YingfanWang/PaCMAP).
    - **LocalMAP:** adjusts the neighborhood graph during optimization to improve local cluster separation. [Paper](https://arxiv.org/abs/2412.15426) (2024; AAAI 2025); implemented in the PaCMAP repository.
- **ParamRepulsor:** hard-negative mining and stronger repulsion address loss of local detail in parametric embeddings. [Paper](https://arxiv.org/abs/2411.15894) (NeurIPS 2024); [code](https://github.com/hyhuang00/ParamRepulsor).
- **New observations:** [ordinary UMAP](https://umap-learn.readthedocs.io/en/latest/transform.html) and [openTSNE](https://opentsne.readthedocs.io/en/stable/api/index.html#openTSNE.TSNEEmbedding.transform) place new points relative to a fitted embedding. Parametric UMAP instead evaluates a learned encoder; new-point support alone does not imply a neural mapping.

## Learned representations

### Reconstruction and masking

- **Autoencoders:** encoder–decoder learning through reconstruction. [Hinton and Salakhutdinov](https://doi.org/10.1126/science.1127647) (2006).
    - **Denoising:** clean-input reconstruction from corrupted observations. [Vincent et al.](https://jmlr.org/papers/v11/vincent10a.html) (2010).
    - **Variational:** probabilistic latent-variable modeling with an approximate posterior. [Kingma and Welling](https://arxiv.org/abs/1312.6114) (2013; ICLR 2014).
    - **Masked:** reconstructs hidden image patches from visible ones. [MAE](https://arxiv.org/abs/2111.06377) (2021; CVPR 2022); [code](https://github.com/facebookresearch/mae).

### Contrastive learning

- **SimCLR:** contrasts augmented image views using a nonlinear projection head; downstream features come from the encoder before that head. [Paper](https://arxiv.org/abs/2002.05709) (ICML 2020); [code](https://github.com/google-research/simclr).
- **[Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)** — 2018. Predicts future latent representations from autoregressive context using a contrastive loss and negative samples; learns sequential features without reconstructing raw observations.
- **[Alignment and uniformity](https://proceedings.mlr.press/v119/wang20k.html)** — 2020. Relates contrastive learning to positive-pair alignment and uniform spread on the unit sphere. The analysis is asymptotic; normalization alone does not guarantee uniform learned features.

### Self-distillation and redundancy reduction

- **BYOL:** an online network predicts a moving target network's representation without explicit negative pairs. [Paper](https://arxiv.org/abs/2006.07733) (NeurIPS 2020).
- **DINO:** self-distillation without labels using a momentum teacher; the precursor to [DINOv2/v3](#images-and-video). [Paper](https://arxiv.org/abs/2104.14294) (ICCV 2021).
- **Barlow Twins:** drives cross-correlation between two augmented views toward identity, combining agreement with redundancy reduction. This objective also underlies [TLDR](#representation-compression). [Paper](https://arxiv.org/abs/2103.03230) (ICML 2021).
- **VICReg:** separates view agreement, per-feature variance control, and covariance regularization. Unlike Barlow Twins' cross-correlation objective, it explicitly constrains each branch's variance. [Paper](https://arxiv.org/abs/2105.04906) (2021; ICLR 2022); [code](https://github.com/facebookresearch/vicreg).

### Prediction in representation space

- **I-JEPA:** predicts representations of masked image regions from visible context, rather than reconstructing pixels as in MAE. [Paper](https://arxiv.org/abs/2301.08243) (CVPR 2023).
- **LeJEPA:** combines a joint-embedding predictive objective with sketched isotropic Gaussian regularization of embeddings. [Paper](https://arxiv.org/abs/2511.08544) (2025, arXiv).
- **Spectral Ghost:** develops a spectral account of representation sufficiency connecting component analysis and self-supervised objectives. [Paper](https://arxiv.org/abs/2601.20154) (2026, arXiv).

## Identifiability and disentanglement

- **[Limits of unsupervised disentanglement](https://arxiv.org/abs/1811.12359)** — 2018/2019. Locatello et al. show that recovering independent factors requires inductive assumptions. Across seven datasets, stronger disentanglement does not consistently reduce downstream sample needs; factor recovery is a distinct goal.
- **[Identifiable VAE / nonlinear ICA](https://arxiv.org/abs/1907.04809)** — 2019/2020. Conditions a factorized latent prior on an observed auxiliary variable. Identifiability holds up to specified transformations under variation and regularity conditions; an ordinary VAE with an unconditional prior lacks this guarantee.

## Extraction by modality

### Images and video

- **DINOv2:** scales self-supervised learning to produce reusable image-level and patch-level features. [Paper](https://arxiv.org/abs/2304.07193) (2023); [code](https://github.com/facebookresearch/dinov2).
- **DINOv3:** Gram anchoring addresses degradation of dense features during extended self-supervised training. [Paper](https://arxiv.org/abs/2508.10104) (2025); [code](https://github.com/facebookresearch/dinov3).
- **V-JEPA 2.1:** dense prediction over visible and masked tokens, plus intermediate-layer self-supervision, extends JEPA features to spatially structured image/video representations. [Report](https://arxiv.org/abs/2603.14482) (2026); [code](https://github.com/facebookresearch/vjepa2).
- **FINO:** adapts visual representations to scientific domains using discrete or continuous metadata alongside self-supervised learning. [Paper](https://arxiv.org/abs/2606.05107) (2026, arXiv).

### Text and retrieval

#### Sentence and document vectors

- **Sentence-BERT:** siamese/triplet training makes BERT sentence vectors usable with cosine similarity, avoiding joint query–document encoding. [Paper](https://arxiv.org/abs/1908.10084) (EMNLP 2019); [code](https://github.com/UKPLab/sentence-transformers).
- **E5:** contrastive pretraining on weakly supervised text pairs; evaluated both without labeled-data fine-tuning and after supervised fine-tuning. [Paper](https://arxiv.org/abs/2212.03533) (2022); [code](https://github.com/microsoft/unilm/tree/master/e5).
- **Qwen3 Embedding:** adapts language-model backbones for instruction-aware multilingual text embeddings. [Paper](https://arxiv.org/abs/2506.05176) (2025); [code](https://github.com/QwenLM/Qwen3-Embedding).

#### Document context and chunk pooling

- **Late chunking:** encodes a long context before mean-pooling individual spans; stores one context-informed vector per chunk. Conventional chunking encodes each chunk separately. [Paper](https://arxiv.org/abs/2409.04701) (2024).
- **pplx-embed-context:** combines diffusion-pretrained bidirectional encoders, contrastive learning, and late chunking for document-context-aware passage vectors. [Report](https://arxiv.org/abs/2602.11151) (2026).

#### Sparse and multivector scoring

- **ColBERT:** independently encodes queries and passages into token vectors; sums each query token's maximum similarity to a passage token. Late interaction retains multiple vectors, whereas late chunking pools each chunk. [Paper](https://arxiv.org/abs/2004.12832) (SIGIR 2020); [code](https://github.com/stanford-futuredata/ColBERT).
- **BGE-M3:** jointly supports dense vectors, sparse lexical weights, and token multivectors through self-knowledge distillation. [Paper](https://arxiv.org/abs/2402.03216) (2024); [code](https://github.com/FlagOpen/FlagEmbedding).
- **Multivector expressivity:** proves a worst-case representation-size separation between Chamfer similarity and single-vector inner products; this is not an empirical retrieval ranking. [Jayaram](https://arxiv.org/abs/2606.23475) (2026, arXiv).

### Audio and time series

- **TS2Vec:** contrastive learning of contextual timestamp features, pooled for subsequences or whole series. [Paper](https://arxiv.org/abs/2106.10466) (2021; AAAI 2022); [code](https://github.com/zhihanyue/ts2vec).
- **MOMENT:** masked time-series pretraining supports classification, forecasting, imputation, and anomaly detection. [Paper](https://arxiv.org/abs/2402.03885) (ICML 2024); [code](https://github.com/moment-timeseries-foundation-model/moment).
- **MantisV2:** synthetic-data pretraining combined with layer selection and token aggregation for frozen time-series features. [Paper](https://arxiv.org/abs/2602.17868) (2026, arXiv); [code](https://github.com/vfeofanov/mantis).
- **BEATs:** iterates acoustic tokenization and masked audio-token prediction to learn general audio representations. [Paper](https://proceedings.mlr.press/v202/chen23ag.html) (ICML 2023); [code](https://github.com/microsoft/unilm/tree/master/beats).
- **[wav2vec 2.0](https://arxiv.org/abs/2006.11477)** — 2020. Masks latent speech representations and predicts jointly learned quantized targets with a contrastive loss. Unlabeled audio pretraining precedes supervised speech-recognition fine-tuning.

### Cross-modal representations

- **CLIP:** contrastive image–text alignment using natural-language supervision. [Paper](https://arxiv.org/abs/2103.00020) (ICML 2021); [code](https://github.com/openai/CLIP).
- **SigLIP 2:** combines sigmoid image–text learning with self-distillation and masked prediction for multilingual alignment and dense visual features. [Paper](https://arxiv.org/abs/2502.14786) (2025); [code](https://github.com/google-research/big_vision).
- **Qwen3-VL-Embedding:** shared vectors for text, images, document images, and video. The accompanying reranker scores pairs rather than producing independently indexable vectors. [Report](https://arxiv.org/abs/2601.04720) (2026); [code](https://github.com/QwenLM/Qwen3-VL-Embedding).
- **[CLAP](https://arxiv.org/abs/2206.04769)** — 2022. Contrastively trains audio and text encoders on paired recordings and descriptions. Their shared space supports audio retrieval and zero-shot classification through text prompts.

## Representation compression

- **Matryoshka representation learning:** trains nested vector prefixes at multiple dimensions. Compression is built into representation training; arbitrary truncation of ordinary embeddings is not equivalent. [Paper](https://arxiv.org/abs/2205.13147) (NeurIPS 2022); [code](https://github.com/RAIVNLab/MRL).
- **TLDR:** learns compression of existing features from neighbor pairs using the Barlow Twins redundancy-reduction objective. PCA instead maximizes retained variance. [Paper](https://arxiv.org/abs/2110.09455) (2021; TMLR 2022); [code](https://github.com/naver/tldr).
- **Compression evidence:** TLDR authors report +4.1 mAP points over whitened PCA at 128 dimensions on ROxford, averaging Medium/Hard protocols, with GeM-AP features and GLD-v2 fitting data. [Table 1; §3.2.1](https://arxiv.org/html/2110.09455v2#S3.SS2.SSS1).

## Evaluation and comparative evidence

### Benchmarks and geometry

- **MTEB / MMTEB:** text-embedding evaluation across tasks, domains, and languages. [MTEB](https://arxiv.org/abs/2210.07316) (2022; EACL 2023); [MMTEB](https://arxiv.org/abs/2502.13595) (2025); [code](https://github.com/embeddings-benchmark/mteb).
- **Neighborhood preservation:** [trustworthiness](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html) penalizes introduced neighbors; [continuity](https://doi.org/10.1016/j.neunet.2006.05.014) penalizes lost neighbors. These measure geometry rather than held-out task performance.
- **Intrinsic dimension:** global and local estimators characterize data geometry; their estimates are not task-optimized embedding dimensions. [Scikit-dimension](https://arxiv.org/abs/2109.02596) (2021); [implementation](https://scikit-dimension.readthedocs.io/).
- **Local versus global structure:** Huang et al. (2022) report stronger neighborhood preservation for t-SNE/art-SNE, but stronger distance-rank preservation for PCA/TriMap/PaCMAP/ForceAtlas2 on most tested datasets. [Comparison, Fig. 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC9296444/).
- **Stability:** Huang et al. (2022) test parameter/preprocessing sensitivity and report five-run variation. Its local/global rankings concern 2D layouts, including transcriptomic data, rather than general predictive features. [Methods and results](https://pmc.ncbi.nlm.nih.gov/articles/PMC9296444/).

### Extraction choices

- **Projection head:** SimCLR's ImageNet linear evaluation favors encoder features before the contrastive projection head over its output. The loss-space representation is not necessarily the best downstream feature. [§4.2](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf#page=6).
- **Context:** late-chunking authors report mean nDCG@10 (0–100) rising from 52.2 to 54.0 across three encoders and four BEIR datasets, using 256-token chunks. Individual settings include ties and regressions. [Table 2](https://arxiv.org/html/2409.04701v3#S4.T2).
- **Layer selection:** MantisV2’s UCR ablations favor intermediate frozen features after more pretraining updates; the 100K-sample, 100-epoch setting favors the final layer. Fine-tuning favors all layers. [§3.3; Appendix B.4](https://arxiv.org/html/2602.17868v1#S3.SS3).

## Software

- **Python — reduction and dimension estimation:** [scikit-learn](https://scikit-learn.org/stable/), [umap-learn](https://umap-learn.readthedocs.io/), [openTSNE](https://opentsne.readthedocs.io/), [PaCMAP/LocalMAP](https://github.com/YingfanWang/PaCMAP), [PHATE](https://github.com/KrishnaswamyLab/PHATE), [scikit-dimension](https://scikit-dimension.readthedocs.io/).
- **Python — extraction:** [Sentence Transformers](https://www.sbert.net/), [timm](https://huggingface.co/docs/timm/feature_extraction), [Kymatio](https://www.kymat.io/), [tsfresh](https://tsfresh.readthedocs.io/); evaluation: [MTEB](https://github.com/embeddings-benchmark/mteb).
- **R:** [dimRed](https://cran.r-project.org/package=dimRed), [dyndimred](https://cran.r-project.org/package=dyndimred), [intrinsicDimension](https://cran.r-project.org/package=intrinsicDimension), [Rdimtools](https://cran.r-project.org/package=Rdimtools) ([paper](https://arxiv.org/abs/2005.11107)); discriminant analysis: [MASS](https://cran.r-project.org/package=MASS), [sda](https://cran.r-project.org/package=sda).
- **JavaScript — extraction:** [Transformers.js](https://huggingface.co/docs/transformers.js/index) runs compatible pretrained models through ONNX Runtime with WASM/WebGPU.
- **JavaScript — visualization:** [tsne](https://www.npmjs.com/package/tsne), [umap-js](https://www.npmjs.com/package/umap-js), [dimred](https://www.npmjs.com/package/dimred); interactive tool: [StatSim Vis](https://statsim.com/vis/).
- **C++:** [Tapkee](https://github.com/lisitsyn/tapkee).

## Related topics

[Feature selection](https://mlpapers.org/feature-selection/) · [Neural networks](https://mlpapers.org/neural-nets/) · [Multiview learning](https://github.com/mlpapers/multiview-learning) · [Interpretability](https://mlpapers.org/interpretability/).

- **Sparse autoencoders for interpretability:** overcomplete sparse features from internal model activations, evaluated for reconstruction and interpretability rather than compactness alone. [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) (2024).
