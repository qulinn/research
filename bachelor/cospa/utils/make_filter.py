import copy
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def get_filter_color(preds):    
    """ Get filter color of image pixel corresponding to prediction values. 

    Args:
        pred (float): prediction value

    Returns:
        binary_px (image): pixel of binary filter
        color_px (image): pixel of color filter
    """
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Color probability Mask 
    pred_probs = sigmoid(preds) # (matplot colormap accepts value [0,1]or[0,255])
    
    # Make color filter
    cm = plt.get_cmap("bwr")

    colors = np.array(cm(pred_probs),dtype=np.float32) # get filter color
    colors = colors[...,:-1] # RGBA -> RGB

    return colors


def get_patchs_from_image(x,patch_size,padding,stride):

    height,width = x.shape[:2]

    patch_imgs = np.array([
        x[i:i+patch_size,j:j+patch_size] \
            for i in range(0,height-2*padding,stride) \
            for j in range(0,width-2*padding,stride)], dtype=np.float32)
    
    patch_imgs_height = np.ceil(((height-2*padding)/stride)).astype("int")
    patch_imgs_width = np.ceil(((width-2*padding)/stride)).astype("int")

    return patch_imgs, patch_imgs_height, patch_imgs_width


def make_pred_result_array(x,preds,padding,stride):

    ret = np.zeros(x.shape[:2], dtype=np.float32)

    for i in range(preds.shape[0]): 
        for j in range(preds.shape[1]):
            ret[
                padding+(i)*stride : padding+(i+1)*stride,
                padding+(j)*stride : padding+(j+1)*stride] = preds[i,j]
    return ret


def paddingCorners(arr, padding):

        arr_ret = np.zeros(arr.shape, dtype=np.float32)*(-10)
        arr_ret[padding:-padding, padding:-padding] = arr[padding:-padding, padding:-padding]

        return arr_ret


def patch_segmentation(img, model, patch_size, stride):
    ''' Color filtering for each image based on predictive probability for each pixel.
        
    Args:
        img (image): image data
        model (keras model): PU Learning pre-trained model
        patch_size (int): Size of patch image
        stride (int): Stride of raster scan
    
    Returns:
        filter_px (image): 
        x_binary_filter (image): binary filter image
        x_color_filter (image): color filter image
        x_superimpose (image): image superimposed on color fiilter.
    '''
    assert img.dtype=="uint8", 'Only accept uint8 data type image.' 

    # Preparation of images.
    img = img.astype('float32')/255.
    x = copy.deepcopy(img)

    # Make a patch images with a raster scan by stride px.
    height,width,_ = x.shape
    padding = patch_size//2
    
    x_binary_filter = np.zeros((height,width,1), dtype=np.uint8) # binary filter
    x_color_filter = np.zeros(x.shape, dtype=np.float32) # color filter
    x_prob_map = np.zeros((height,width,1),dtype=np.float32) # probability map
    x_superimpose = copy.deepcopy(x) # result of segmentation (a filtered image)

    # Make a patch images with a raster scan by stride px.
    patch_imgs, patch_imgs_height, patch_imgs_width = get_patchs_from_image(x,patch_size,padding,stride)
    print("patch_imgs", patch_imgs.shape)
    print("model.input_shape", model.input_shape)

    # Prediction.
    # preds = model.predict(patch_imgs, batch_size=512)

    preds = model.predict_on_batch(patch_imgs)
    preds = preds.reshape((patch_imgs_height,patch_imgs_width))

    # get average prediction values
    x_prob_map = make_pred_result_array(x,preds,padding,stride)
    x_prob_map = paddingCorners(x_prob_map,padding)

    # binary_px, color_px = get_filter_color(preds[i,j])
    x_prob_map = np.round(x_prob_map, decimals=3)
    x_binary_filter = np.where(x_prob_map<=0,0,255).astype(np.uint8)
    x_color_filter = get_filter_color(x_prob_map).astype(np.float32)
    
    # Superimpose
    alpha, beta = 0.7, 0.3
    # x_superimpose = cv2.addWeighted(x_superimpose, alpha, x_color_filter, beta, 0, dtype=cv2.CV_32F)
    x_superimpose = cv2.addWeighted(x_superimpose, alpha, x_color_filter, beta, 0)

    return x_prob_map, x_binary_filter, x_color_filter, x_superimpose


def show_segmentation_result(x, x_color_filter, x_superimpose):
    """ Make a display figure of segmentation result images.

    Args:
        x (image): original image.
        x_filter (image): color filter image.
        x_result (image): original image superinpsed on color filter image.

    Return:
        figure (matplot figure): 1×3 image window figure.
    """

    figure = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(wspace=0.1, hspace=-0.15)
    gs_master = GridSpec(nrows=2, ncols=3)

    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])
    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 1])
    gs_3 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 2])
    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_master[1, :])

    axes_1 = figure.add_subplot(gs_1[:, :])
    axes_2 = figure.add_subplot(gs_2[:, :])
    axes_3 = figure.add_subplot(gs_3[:, :])
    axes_4 = figure.add_subplot(gs_4[:, :])

    axes_1.imshow(x)
    axes_1.set_title("Original", fontsize=12)
    axes_1.axis("off")

    axes_2.imshow(x_color_filter)
    axes_2.set_title("Probability", fontsize=12)
    axes_2.axis("off")

    axes_3.imshow(x_superimpose)
    axes_3.set_title("Segmented", fontsize=12)
    axes_3.axis("off")

    # Color Bar
    cbar = axes_4.figure.colorbar(
                cm.ScalarMappable(cmap='bwr'), ticks=[0, 1], # norm=norm ),
                ax=axes_4, orientation='horizontal', fraction=0.9, pad=0.05, shrink=0.9, aspect=40) #, extend='both')
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_xticklabels(['Negative',  'Positive'])
    cbar.set_label("Probability colorbar",size=14,labelpad=-10)
    axes_4.axis('off')

    return figure


def show_CRF_result(x, x_before, x_after):
    """ Make a display figure of CRF result images.

    Args:
        x (image): original image.
        x_before (image): original binary segmenation image.
        x_after (image): binary segmenation image after appling CRF.

    Return:
        figure (matplot figure): 1×3 image window figure.
    """

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.1, hspace=-0.15)

    fig.suptitle('Result of CRF', fontsize=16)
    fig.subplots_adjust(top=1.25)

    axs[0].imshow(x)
    axs[0].set_title("Original", fontsize=12)
    axs[0].axis("off")

    x_before = cv2.cvtColor(x_before, cv2.COLOR_GRAY2RGB)
    axs[1].imshow(x_before)
    axs[1].set_title("CRF not applied", fontsize=12)
    axs[1].axis("off")

    axs[2].imshow(x_after)
    axs[2].set_title("CRF applied", fontsize=12)
    axs[2].axis("off")

    return fig