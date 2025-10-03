import numpy as np
import numba
import sklearn.datasets
import matplotlib.pyplot as plt
import umap

# color_map = {
#     1: '#e50000',
#     2: '#F72585',
#     3: '#B5179E',
#     4: '#7209B7',
#     5: '#4361EE',
# }
color_map = {
    1: '#e50000',
    2: '#B5179E',
    3: '#7209B7',
    4: '#4361EE',
}
print("Loading data and label ...")
qgcn_embedding = np.load("./output/ablation/QGCN_embeddings.npy") # Euclidean
qgcn2_embedding = np.load("./output/ablation/QGCN2_embeddings.npy") # Poincare
qgt_embedding = np.load("./output/ablation/QGT_embeddings.npy") # Poincare

label = np.load("./output/nc/GCN/cora/1234/degrees.npy") # degree label
bins =  [1, 5.1, 10.1, 20.1, 1000] #  Level 1 = 1068 + 1223, Level 2 = 321, Level 3 = 72, Level 4 = 24
label_bins = np.digitize(label, bins=bins[1:], right=False)

mapper1 = umap.UMAP(output_metric="euclidean", random_state=1234)
mapper2 = umap.UMAP(output_metric="hyperboloid", random_state=1234)
mapper3 = umap.UMAP(output_metric="hyperboloid", random_state=1234)
print("Done !")

# plot QGCN
print("UMAP for QGCN !")
umap1_embedding = mapper1.fit_transform(qgcn_embedding)
x1, y1 = umap1_embedding[:, 0], umap1_embedding[:, 1]
print("Done !")

print("Plotting QGCN !")
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(5, 5))
for i in range(4):
    idx = label_bins == i
    plt.scatter(x1[idx], y1[idx], s=6, color=color_map[i+1])
plt.title("$\mathcal{Q}$-GCN embedding on cora")
plt.tight_layout()
plt.xticks([])  
plt.yticks([])  
plt.savefig("./output/ablation/QGCN_embeddings_plot.pdf", dpi=fig.dpi)
print("Done !\n")

# plot QGCN2
print("UMAP for QGCN2 !")
umap2_embedding = mapper2.fit_transform(qgcn2_embedding)
x2, y2 = umap2_embedding[:, 0], umap2_embedding[:, 1]
print("Done !")

print("Plotting QGCN2 !")
z2 = (1 + x2 ** 2 + y2 ** 2) ** 0.5
disk_x2 = x2 / (1 + z2)
disk_y2 = y2 / (1 + z2)
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
boundary = plt.Circle((0, 0), 1, color='black', fill=False)
for i in range(4):
    idx = label_bins == i
    ax.scatter(disk_x2[idx], disk_y2[idx], s=6, color=color_map[i+1])
ax.add_artist(boundary)
ax.axis('off')
plt.title("$\mathcal{Q}$-GCN2 embedding on cora")
plt.tight_layout()
plt.xticks([])  
plt.yticks([])  
plt.savefig("./output/ablation/QGCN2_embeddings_plot.pdf", dpi=fig.dpi)
print("Done !\n")

# plot QGT
print("UMAP for QGT !")
umap3_embedding = mapper3.fit_transform(qgt_embedding)
x3, y3 = umap3_embedding[:, 0], umap3_embedding[:, 1]
print("Done !")

print("Plotting QGT !")
z3 = (1 + x3 ** 2 + y3 ** 2) ** 0.5
disk_x3 = x3 / (1 + z3)
disk_y3 = y3 / (1 + z3)
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
boundary = plt.Circle((0, 0), 1, color='black', fill=False)
for i in range(4):
    idx = label_bins == i
    ax.scatter(disk_x3[idx], disk_y3[idx], s=6, color=color_map[i+1])
ax.add_artist(boundary)
ax.axis('off')
plt.title("$\mathcal{Q}$-GT embedding on cora")
plt.tight_layout()
plt.xticks([])  
plt.yticks([])  
plt.savefig("./output/ablation/QGT_embeddings_plot.pdf", dpi=fig.dpi)
print("Done !\n")