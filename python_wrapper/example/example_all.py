"""
Same as example.py but here we create embeddings for three metrics and plot them all together

"""
import SpidrWrapper as spidr
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
from os.path import dirname as up
from scipy.sparse import csr_matrix
from nptsne import TextureTsne as TSNE  # Texture refers to implementation details, not texture-aware DR
from umap import UMAP
from sklearn.manifold import MDS

from utils import load_binary, assign_embedding_colors

# load data
data_path = up(up(getcwd())) + "\\example\\data\\"
data_name = "CheckeredBoxes_2Ch_32.bin"
data_raw = load_binary(data_path + data_name)

# prep data
data = data_raw.reshape((-1, 2))
data_glob_ids = np.arange(data.shape[0])
imgWidth = int(np.sqrt(data.shape[0]))
imgHeight = imgWidth
numPoints = data.shape[0]
data_img = data.reshape((imgHeight, imgWidth, 2))

# settings
sp_metrics = [spidr.DistMetric.Chamfer_pc, spidr.DistMetric.QF_hist, spidr.DistMetric.Bhattacharyya]
sp_weight = spidr.WeightLoc.uniform
sp_neighborhoodSize = 1  # one neighbor in each direction, i.e. a 3x3 neighborhood

#################################
# spatially informed embeddings #
#################################
embs_tsne_sp = {}
embs_umap_sp = {}
embs_mds_sp = {}

for sp_metric in sp_metrics:
    print(f"Metric: {sp_metric}")
    #########
    # t-SNE #
    #########
    print("# Texture-aware t-SNE with HDILib (nptsne)")
    # instantiate spidrlib
    alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight, numHistBins=5,
                                    numLocNeighbors=sp_neighborhoodSize, aknnAlgType=spidr.KnnAlgorithm.hnsw)

    # embed with t-SNE
    embs_tsne_sp[sp_metric] = alg_spidr.fit_transform(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)

    ########
    # UMAP #
    ########
    print("# Texture-aware UMAP with umap-learn")
    # instantiate spidrlib
    alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight, numHistBins=5,
                                    numLocNeighbors=sp_neighborhoodSize, aknnAlgType=spidr.KnnAlgorithm.hnsw)
    nn = alg_spidr.nn

    # get knn dists to compute umap
    knn_ind, knn_dists = alg_spidr.fit(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)

    # create sparse matrix with scipy
    knn_ind = np.array(knn_ind)  # .reshape((numPoints, nn))
    knn_dists = np.array(knn_dists)  # .reshape((numPoints, nn))
    knn_ind_row = np.repeat(np.arange(0, numPoints), nn)  # .reshape((numPoints, nn))

    knn_csr = csr_matrix((knn_dists, (knn_ind_row, knn_ind)), shape=(numPoints, numPoints))

    # embed with umap
    alg_umap = UMAP()
    embs_umap_sp[sp_metric] = alg_umap.fit_transform(knn_csr)


    #######
    # MDS #
    #######
    print("# Texture-aware MDS with scikit-learn")
    # instantiate spidrlib
    alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight, numHistBins=5,
                                    numLocNeighbors=sp_neighborhoodSize, aknnAlgType=spidr.KnnAlgorithm.full_dist_matrix)

    # get full dist matrix to compute mds
    _, knn_dists = alg_spidr.fit(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)

    # create full distance matrix
    knn_dists = np.array(knn_dists).reshape((numPoints, numPoints))

    # embed with MDS
    alg_mds = MDS(dissimilarity='precomputed', n_jobs=-1)
    embs_mds_sp[sp_metric] = alg_mds.fit_transform(knn_dists)


#######################
# standard embeddings #
#######################

# standard t-SNE
print("# Standard t-SNE with HDILib (nptsne)")
alg_tsne = TSNE()
emb_tsne_std = alg_tsne.fit_transform(data).reshape((numPoints, 2))

# standard UMAP
print("# Standard UMAP with umap-learn")
alg_umap = UMAP()
emb_umap_std = alg_umap.fit_transform(data)

# standard MDS
print("# Standard MDS with scikit-learn")
alg_mds = MDS(dissimilarity='euclidean', n_jobs=-1)
emb_mds_std = alg_mds.fit_transform(data)


####################
# Plots embeddings #
####################

# map embedding positions to colors and then back to the image space
clm_path = '../../example/eval/2d_Mittelstaed.png'

embs_tsne_sp_colors = {}
embs_umap_sp_colors = {}
embs_mds_sp_colors = {}

for sp_metric in sp_metrics:
    embs_tsne_sp_colors[sp_metric] = assign_embedding_colors(embs_tsne_sp[sp_metric], clm_path, rot90=3)
    embs_umap_sp_colors[sp_metric] = assign_embedding_colors(embs_umap_sp[sp_metric], clm_path, rot90=3)
    embs_mds_sp_colors[sp_metric] = assign_embedding_colors(embs_mds_sp[sp_metric], clm_path, rot90=3)

emb_umap_std_colors = assign_embedding_colors(emb_umap_std, clm_path, rot90=3)
emb_tsne_std_colors = assign_embedding_colors(emb_tsne_std, clm_path, rot90=3)
emb_mds_std_colors = assign_embedding_colors(emb_mds_std, clm_path, rot90=3)

# Plot embedding
fig, axs = plt.subplots(3, 8, figsize=(15, 5))
#fig.suptitle('Embeddings and data colored based on embeddings')


def pltColProj(row_n, col_n, title, emb, emb_cols):
    # emb scatter
    #axs[row_n, col_n].title.set_text(title)
    axs[row_n, col_n].scatter(emb[:, 0], emb[:, 1], c=emb_cols, s=5, alpha=0.5)
    axs[row_n, col_n].get_xaxis().set_visible(False)
    axs[row_n, col_n].get_yaxis().set_visible(False)
    # img re-colored
    axs[row_n, col_n+1].imshow(emb_cols.reshape((imgHeight, imgWidth, 3)), aspect="auto")
    axs[row_n, col_n+1].xaxis.tick_top()
    axs[row_n, col_n+1].get_xaxis().set_visible(False)
    axs[row_n, col_n+1].get_yaxis().set_visible(False)


for metric_id, sp_metric in enumerate(sp_metrics):
    col_id = 2 + 2*metric_id
    pltColProj(0, col_id, f't-SNE w/ {sp_metric.name}', embs_tsne_sp[sp_metric], embs_tsne_sp_colors[sp_metric])
    pltColProj(1, col_id, f'UMAP w/ {sp_metric.name}', embs_umap_sp[sp_metric], embs_umap_sp_colors[sp_metric])
    pltColProj(2, col_id, f'MDS w/ {sp_metric.name}', embs_mds_sp[sp_metric], embs_mds_sp_colors[sp_metric])

pltColProj(0, 0, 't-SNE std', emb_tsne_std, emb_tsne_std_colors)
pltColProj(1, 0, 'UMAP std', emb_umap_std, emb_umap_std_colors)
pltColProj(2, 0, 'MDS std', emb_mds_std, emb_mds_std_colors)

# label rows
pad = 5
axs[0, 0].annotate("t-SNE", xy=(0, 0), xytext=(-65, 0), size=10)    # xytext depends on scatterplot ranges
axs[1, 0].annotate("UMAP", xy=(0, 0), xytext=(-20, 2), size=10)     # automating this would be better
axs[2, 0].annotate("MDS", xy=(0, 0), xytext=(-1.5, -.1), size=10)

# label columns
height_lab_c = 0.91
plt.figtext(0.22, height_lab_c, "Standard", va="center", ha="center", size=10)
plt.figtext(0.41, height_lab_c, "Chamfer point cloud", va="center", ha="center", size=10)
plt.figtext(0.61, height_lab_c, "Histograms", va="center", ha="center", size=10)
plt.figtext(0.81, height_lab_c, "Bhattacharyya", va="center", ha="center", size=10)

#plt.tight_layout()
#plt.show()
plt.savefig("example_several_DR_and_metrics.pdf", format="pdf", bbox_inches="tight")

##################
# Plots the data #
##################

data_min = np.min(data_img)
data_max = np.max(data_img)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#fig.suptitle('Data channels and embeddings')

axs[0].title.set_text('Channel 1')
im1 = axs[0].imshow(data_img[:, :, 0], aspect="auto", vmin=data_min, vmax=data_max)
axs[1].title.set_text('Channel 2')
im2 = axs[1].imshow(data_img[:, :, 1], aspect="auto", vmin=data_min, vmax=data_max)

fig.colorbar(im2, ax=axs.ravel().tolist())

#plt.tight_layout()
plt.savefig("example_data_channels.pdf", format="pdf", bbox_inches="tight")

