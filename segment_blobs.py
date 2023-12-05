# %%
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

DATA_HOME = Path('/storage/data/Chaoyi/Microscope images')
N_CLUSTERS = 3

sns.set(
    context="talk",
    style="ticks",
    font="Arial",
    font_scale=1.0,
    rc={"svg.fonttype": "none", "lines.linewidth": 1.6, "figure.autolayout": True},
)

# %%
sample_image_files = (DATA_HOME / 'ZCY-014' / 'test').glob('*.jpg')
sample_image_files = {p.stem: p for p in sample_image_files}
sample_image_files

# %%
#read image file
imgs_BGR = {id: cv2.imread(str(path)) for id, path in sample_image_files.items()}
imgs_RGB = {id: cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for id, img in imgs_BGR.items()}
imgs_small_RGB = {id: cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)) for id, img in imgs_RGB.items()}
imgs_HSV = {id: cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for id, img in imgs_BGR.items()}
imgs_small_HSV = {id: cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)) for id, img in imgs_HSV.items()}
imgs_LAB = {id: cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for id, img in imgs_BGR.items()}
imgs_small_LAB = {id: cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)) for id, img in imgs_LAB.items()}

# %%
n_rows=len(imgs_small_RGB)
fig, axs = plt.subplots(nrows=n_rows, figsize=(5, n_rows * 5))
for (id, data), ax in zip(imgs_small_RGB.items(), axs):
    ax.imshow(data)
    ax.set_title(id)

# %%
best_sample_id = 'ZCY-014c 2023-08-29_18-21-58'

# %%
plt.imshow(np.prod(imgs_small_HSV[best_sample_id][..., 2:], axis=-1))
plt.figure()
plt.imshow(imgs_small_RGB[best_sample_id])

# %%
# generate the mask from Segment Anything
sam = sam_model_registry['default'](checkpoint='../sam_vit_h_4b8939.pth')
sam.to('cuda')
mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16, points_per_side=64, box_nms_thresh=0.1)

# %%
masks = {id: mask_generator.generate(img) for id, img in tqdm(imgs_small_RGB.items())}
{id: len(img_masks) for id, img_masks in masks.items()}

# %%
masks = {id: [m for m in img_masks if m['area'] < 5000] for id, img_masks in masks.items()}

{id: len(img_masks) for id, img_masks in masks.items()}

# %% [markdown]
# # Choosing representative colours for each droplet

# %% [markdown]
# ## Option 1: Using the mean colour of each droplet
# 
# Pros:
# 
# Cons:
# 

# %%
def avg_region_color(img, masks):
    mask_pixels = [img[m['segmentation']] for m in masks]
    region_pixels = np.array([m.mean(axis=0) for m in mask_pixels])
    return region_pixels / 255

# %%
def max_region_color(img, masks):
    """
    Return the color of the pixel with the largest norm in each mask.
    Meant for use with RGB images only.
    """
    mask_pixels = [img[m['segmentation']] for m in masks]
    representative_pixel = [np.linalg.norm(pixels, axis=-1).argmax() for pixels in mask_pixels]
    region_pixels = np.array([mask_pixels[i][representative_pixel[i]] for i in range(len(mask_pixels))])
    return region_pixels / 255

# %%
def max_value_region_color(img, masks):
    """
    Return the color of the pixel with the largest V (value) in each mask.
    Meant for use with HSV images only.
    """
    mask_pixels = [img[m['segmentation']] for m in masks]
    # find the pixel with the largest value in the third (V) channel
    representative_pixel = [pixels[..., 2].argmax() for pixels in mask_pixels]
    region_pixels = np.array([mask_pixels[i][representative_pixel[i]] for i in range(len(mask_pixels))])
    return region_pixels / 255

# %%
def max_sat_region_color(img, masks):
    """
    Return the color of the pixel with the largest S (saturation) in each mask.
    Meant for use with HSV images only.
    """
    mask_pixels = [img[m['segmentation']] for m in masks]
    representative_pixel = [pixels[..., 1].argmax() for pixels in mask_pixels]
    region_pixels = np.array([mask_pixels[i][representative_pixel[i]] for i in range(len(mask_pixels))])
    return region_pixels / 255

# %%
def max_SV_region_color(img, masks):
    """
    Return the color of the pixel with the largest product of S (saturation) and V (value) in each mask.
    Meant for use with HSV images only.
    """
    mask_pixels = [img[m['segmentation']] for m in masks]
    # indices of the middle 25% V pixels
    representative_pixels = [np.argsort(pixels[..., 2])[-(pixels.shape[0] // 4):] for pixels in mask_pixels]
    # index of max S among the top 50% V pixels
    representative_pixel = [pixels[:, 1][representative_pixels[i]].argmax() for i, pixels in enumerate(mask_pixels)]
    region_pixels = np.array([mask_pixels[i][representative_pixels[i][representative_pixel[i]]] for i in range(len(mask_pixels))])
    return region_pixels / 255

# %%
def max_SV_AB_color(img_hsv, img_lab, masks):
    """
    Return the color of the pixel with the largest product of S (saturation) and V (value) in each mask.
    Meant for use with HSV images only.
    """
    mask_pixels_hsv = [img_hsv[m['segmentation']] for m in masks]
    mask_pixels_lab = [img_lab[m['segmentation']] for m in masks]
    # indices of the middle 25% V pixels
    representative_pixels = [np.argsort(pixels[..., 2])[-(pixels.shape[0] // 4):] for pixels in mask_pixels_hsv]
    # index of max S among the top 50% V pixels
    representative_pixel = [pixels[representative_pixels[i]][:, 1].argmax() for i, pixels in enumerate(mask_pixels_hsv)]
    region_pixels = np.array([mask_pixels_lab[i][representative_pixels[i][representative_pixel[i]]] for i in range(len(mask_pixels_hsv))])
    return region_pixels / 255

# %%
def max_SV_RGB_color(img_hsv, img_rgb, masks):
    """
    Return the color of the pixel with the largest product of S (saturation) and V (value) in each mask.
    Meant for use with HSV images only.
    """
    mask_pixels_hsv = [img_hsv[m['segmentation']] for m in masks]
    mask_pixels_rgb = [img_rgb[m['segmentation']] for m in masks]
    # indices of the middle 25% V pixels
    representative_pixels = [np.argsort(pixels[..., 2])[-(pixels.shape[0] // 4):] for pixels in mask_pixels_hsv]
    # index of max S among the top 25% V pixels
    representative_pixel = [pixels[representative_pixels[i]][:, 1].argmax() for i, pixels in enumerate(mask_pixels_hsv)]
    region_pixels = np.array([mask_pixels_rgb[i][representative_pixels[i][representative_pixel[i]]] for i in range(len(mask_pixels_hsv))])
    return region_pixels / 255

# %%
def max_V_RGB_color(img_hsv, img_rgb, masks):
    """
    Return the RGB color of the pixel with the largest V (value) in each mask.
    """
    mask_pixels_hsv = [img_hsv[m['segmentation']] for m in masks]
    mask_pixels_rgb = [img_rgb[m['segmentation']] for m in masks]
    representative_pixel = [pixels[..., 2].argmax() for pixels in mask_pixels_hsv]
    region_pixels = np.array([mask_pixels_rgb[i][pixel] for i, pixel in enumerate(representative_pixel)])
    return region_pixels / 255

# %%
def max_SV_prod_AB_color(img_hsv, img_lab, masks):
    """
    Return the color of the pixel with the largest product of S (saturation) and V (value) in each mask.
    Meant for use with HSV images only.
    """
    mask_pixels_hsv = [img_hsv[m['segmentation']] for m in masks]
    mask_pixels_lab = [img_lab[m['segmentation']] for m in masks]

    representative_pixels = [np.prod(pixels[..., 1:], axis=-1).argmax() for pixels in mask_pixels_hsv]

    region_pixels = np.array([mask_pixels_lab[i][representative_pixels[i]] for i in range(len(mask_pixels_hsv))])
    return region_pixels / 255

# %%
def centre_region_color(img, masks):
    result = []
    for mask in masks:
        h, v = np.nonzero(mask['segmentation'])
        h, v = int(h.mean()), int(v.mean())
        result.append(img[h, v])
    return np.array(result) / 255

# %%
hues = max_SV_region_color(imgs_small_HSV[best_sample_id], masks[best_sample_id])[..., :1]
gm = GaussianMixture(n_components=N_CLUSTERS)
fit = gm.fit(hues)
plt.hist(hues, bins=100, density=True)
mus = fit.means_[:, 0]
sigmas = np.sqrt(fit.covariances_[:, 0, 0])
pdfs = np.exp(-0.5 * ((np.linspace(0, 1, 100)[:, None] - mus) / sigmas) ** 2) / (np.sqrt(2 * np.pi) * sigmas) * fit.weights_[None, :]
plt.plot(np.linspace(0, 1, 100), pdfs.sum(axis=-1))
fit.means_, sigmas, fit.weights_

# %%
hues = max_SV_RGB_color(imgs_small_HSV[best_sample_id], imgs_small_RGB[best_sample_id], masks[best_sample_id])
gm = GaussianMixture(n_components=N_CLUSTERS)
fit = gm.fit(hues)
plt.hist2d(*hues[:, [1, 2]].T, bins=100, density=True)
mus = fit.means_
plt.scatter(*mus[:, [1, 2]].T, c='r')

# %%
region_rgbs = {id: max_SV_RGB_color(imgs_small_HSV[id], imgs_small_RGB[id], masks[id]) for id in tqdm(masks)}
region_hsvs = {id: max_SV_region_color(imgs_small_HSV[id], masks[id]) for id in tqdm(masks)}
region_labs = {id: max_SV_AB_color(imgs_small_HSV[id], imgs_small_LAB[id], masks[id]) for id in tqdm(masks)}

# %%
region_color_tables = {id: pd.DataFrame(np.concatenate([region_rgbs[id], region_hsvs[id], region_labs[id]], axis=1), columns=['R', 'G', 'B', 'H', 'S', 'V', 'l', 'a', 'b'] ) for id in region_rgbs}

# %%
total_table = pd.concat(region_color_tables.values(), keys=region_color_tables.keys())
total_table.rename_axis(['image_id', 'mask_no'])

# %% [markdown]
# ## Step 1: Fit model parameters to the best image

# %%
total_table.loc[best_sample_id]

# %%
gm = GaussianMixture(n_components=3)
# predictors = ['H']
# predictors = ['a', 'b']
predictors = ['R', 'G', 'B']
fit = gm.fit(total_table[predictors].to_numpy())

# %%
total_table.hist(column='H', bins=100, density=True)
mus = fit.means_[:, 0]
sigmas = np.sqrt(fit.covariances_[:, 0, 0])
pdfs = np.exp(-0.5 * ((np.linspace(0, 1, 100)[:, None] - mus) / sigmas) ** 2) / (np.sqrt(2 * np.pi) * sigmas) * fit.weights_[None, :]
plt.plot(np.linspace(0, 1, 100), pdfs.sum(axis=-1))
fit.means_, sigmas, fit.weights_

# %%
total_table['cluster'] = gm.predict(total_table[predictors].to_numpy())
total_table['prob'] = gm.predict_proba(total_table[predictors].to_numpy()).max(axis=-1)
sure_table = total_table.query('prob > 0.95')
sure_table

# %%
sure_table.hist(column='prob',bins=100, density=True)

# %%
plt.hist2d(*sure_table[['R', 'B']].to_numpy().T, bins=100);

# %%
sure_table.loc[best_sample_id].index

# %%
image_ids = list(imgs_small_RGB)
cluster_numbers = list(range(N_CLUSTERS))

n_rows = len(image_ids)
n_cols = N_CLUSTERS

fig, axs = plt.subplots(ncols=n_cols + 1, nrows=n_rows, figsize=((n_cols + 1) * 15, n_rows * 15))

for i, img_id in enumerate(image_ids):
# for i, img_id in enumerate([best_sample_id]):
    mask_ids = sure_table.loc[img_id].index
    image_masks = np.array([masks[img_id][i]['segmentation'] for i in mask_ids])
    img = imgs_small_RGB[img_id]
    axs[i, 0].set_title(img_id)
    colors = sure_table.loc[img_id][['R', 'G', 'B']].to_numpy()
    image_clusters = sure_table.loc[img_id]['cluster'].to_numpy()
    cluster_pixels = colors[:, None, None, :] * image_masks[:, :, :, None]
    for j in cluster_numbers:
        selector = np.where(image_clusters==j)[0]
        # cluster_pixels = sum(img * image_masks[i][..., None] for i in selector)
        # show each region with the average colour
        # axs[i, j].imshow(cluster_pixels[selector].max(axis=0))
        axs[i, j].imshow(cluster_pixels[selector].max(axis=0))
    axs[i, -1].imshow(img)


# %%
image_ids = list(imgs_small_RGB)
cluster_numbers = list(range(N_CLUSTERS))

n_rows = len(image_ids)
n_cols = N_CLUSTERS

for i, img_id in enumerate(image_ids):
# for i, img_id in enumerate([best_sample_id]):
    mask_ids = sure_table.loc[img_id].index
    image_masks = np.array([masks[img_id][i]['segmentation'] for i in mask_ids])
    img = imgs_small_RGB[img_id]
    axs[i, 0].set_title(img_id)
    colors = sure_table.loc[img_id][['R', 'G', 'B']].to_numpy()
    image_clusters = sure_table.loc[img_id]['cluster'].to_numpy()
    cluster_pixels = colors[:, None, None, :] * image_masks[:, :, :, None]
    for j in cluster_numbers:
        fig, ax = plt.subplots(figsize=(5,5))
        plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
        selector = np.where(image_clusters==j)[0]
        # cluster_pixels = sum(img * image_masks[i][..., None] for i in selector)
        # show each region with the average colour
        # axs[i, j].imshow(cluster_pixels[selector].max(axis=0))
        ax.imshow(cluster_pixels[selector].max(axis=0))
        plt.savefig(f"Outcomes-{img_id}-{j}.png", dpi=600, bbox_inches="tight", transparent=True)
        plt.savefig(f"Outcomes-{img_id}-{j}.svg", bbox_inches="tight", transparent=True)


