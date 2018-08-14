from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from data.sensitive_cl3_dataset import SensitiveCl3Dataset

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target

print(X.shape,y.shape)
n_samples, n_features = X.shape
n_neighbors = 30


transform_val_list = [
        transforms.Resize(size=(224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform = transforms.Compose(transform_val_list)
sensitive = SensitiveCl3Dataset(
    '/home/zengh/Dataset/oxy/oxySensitive/Classification/train_cl3.txt', '/home/zengh/Dataset/oxy/oxySensitive/Sensitive_train_img',transform=transform)
length = len(sensitive)
dataloader = DataLoader(sensitive,shuffle=True,
                                  num_workers=4,
                                  batch_size=100)

# for i in range(len(sensitive)):
for i,data in enumerate(dataloader):
    img,label = data
    print(img.shape,label.shape)




#----------------------------------------------------------------------
# Scale and visualize the embedding vectors


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.show()
