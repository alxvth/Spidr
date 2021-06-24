import SpidrWrapper as spidr
import numpy as np
import os
from os.path import dirname as up
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import nptsne
import umap
from utils import load_binary, assign_embedding_colors

# load data
data_path = up(up(os.getcwd())) + "\\example\\data\\"
data_name = "CheckeredBoxes_2Ch_32.bin"
data_raw = load_binary(data_path + data_name)

# prep data
data = data_raw.reshape((-1, 2))
data_glob_ids = np.arange(data.shape[0])
imgWidth = int(np.sqrt(data.shape[0]))
imgHeight = imgWidth
numPoints = data.shape[0]
data_img = data.reshape((imgHeight, imgWidth, 2))

# instantiate spidrlib
spidr = spidr.SpidrAnalysis(distMetric=spidr.DistMetric.Chamfer_pc, kernelType=spidr.WeightLoc.uniform,
                            numLocNeighbors=1, aknnAlgType=spidr.KnnAlgorithm.hnsw)

# embed with t-SNE
emb_tsne = spidr.fit_transform(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)

plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1])
plt.show()

# get knn dists to compute umap
knn_ind, knn_dists = spidr.fit(X=data_img, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)
nn = spidr.nn

# create sparse matrix with scipy
knn_ind = np.array(knn_ind)  # .reshape((numPoints, nn))
knn_dists = np.array(knn_dists)  # .reshape((numPoints, nn))
knn_ind_row = np.repeat(np.arange(0, numPoints), nn)  # .reshape((numPoints, nn))

knn_csr = csr_matrix((knn_dists, (knn_ind_row, knn_ind)), shape=(numPoints, numPoints))

# embed with umap
umapper = umap.UMAP()
emb_umap = umapper.fit_transform(knn_csr)

plt.scatter(emb_umap[:, 0], emb_umap[:, 1])
plt.show()

# compare with standard embedding (t-SNE & UMAP)
tsne = nptsne.TextureTsne()
emb_tsne_std = tsne.fit_transform(data).reshape((numPoints, 2))

emb_umap_std = umapper.fit_transform(data)

# Plot the embeddings and data
fig, axs = plt.subplots(2, 3, figsize=(7.5, 5))

axs[0, 0].title.set_text('Data, ch1')
axs[0, 0].imshow(data_img[:, :, 0])
axs[1, 0].title.set_text('Data, ch2')
axs[1, 0].imshow(data_img[:, :, 1])

axs[0, 1].title.set_text('t-SNE w/ chamfer')
axs[0, 1].scatter(emb_tsne[:, 0], emb_tsne[:, 1])
axs[1, 1].title.set_text('t-SNE std')
axs[1, 1].scatter(emb_tsne_std[:, 0], emb_tsne_std[:, 1])

axs[0, 2].title.set_text('UMAP w/ chamfer')
axs[0, 2].scatter(emb_umap[:, 0], emb_umap[:, 1])
axs[1, 2].title.set_text('UMAP std')
axs[1, 2].scatter(emb_umap_std[:, 0], emb_umap_std[:, 1])

plt.tight_layout()
plt.show()


# map embedding positions to colors and then back to the image space
clm_path = '../../example/eval/2d_Mittelstaed.png'
emb_tsne_colors = assign_embedding_colors(emb_tsne, clm_path, rot90=3)
emb_tsne_std_colors = assign_embedding_colors(emb_tsne_std, clm_path, rot90=3)
emb_umap_colors = assign_embedding_colors(emb_umap, clm_path, rot90=3)
emb_umap_std_colors = assign_embedding_colors(emb_umap_std, clm_path, rot90=3)

# Plot embedding
fig, axs = plt.subplots(2, 4, figsize=(10, 5))

axs[0, 0].title.set_text('t-SNE w/ chamfer')
axs[0, 0].scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=emb_tsne_colors, s=5, alpha=0.5)
axs[1, 0].imshow(emb_tsne_colors.reshape((imgHeight, imgWidth, 3)))

axs[0, 1].title.set_text('t-SNE std')
axs[0, 1].scatter(emb_tsne_std[:, 0], emb_tsne_std[:, 1], c=emb_tsne_std_colors, s=5, alpha=0.5)
axs[1, 1].imshow(emb_tsne_std_colors.reshape((imgHeight, imgWidth, 3)))

axs[0, 2].title.set_text('UMAP w/ chamfer')
axs[0, 2].scatter(emb_umap[:, 0], emb_umap[:, 1], c=emb_umap_colors, s=5, alpha=0.5)
axs[1, 2].imshow(emb_umap_colors.reshape((imgHeight, imgWidth, 3)))

axs[0, 3].title.set_text('UMAP std')
axs[0, 3].scatter(emb_umap_std[:, 0], emb_umap_std[:, 1], c=emb_umap_std_colors, s=5, alpha=0.5)
axs[1, 3].imshow(emb_umap_std_colors.reshape((imgHeight, imgWidth, 3)))

plt.tight_layout()
plt.show()
