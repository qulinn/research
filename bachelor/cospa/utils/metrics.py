import numpy as np

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h,w = segm.shape[0], segm.shape[1]
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def pixel_accuracy(y_pred, y_true):
    ''' Pixel Accuracy for Binary Segmentation (this method can be applied to muitl label image segmentation.)
        sum_i(n_ii) / sum_i(t_i)
    '''
    assert y_pred.shape==y_true.shape,"DiffDim: Different dimensions of matrices!"
    
    px_acc = 0
    for i in range(len(y_pred)):
        eval_segm = y_pred[i]
        gt_segm = y_true[i]

        cl, n_cl = extract_classes(gt_segm)
        eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        sum_n_ii = 0
        sum_t_i  = 0

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            sum_t_i  += np.sum(curr_gt_mask)
    
        if (sum_t_i == 0):
            px_acc += 0
        else:
            px_acc += sum_n_ii / sum_t_i

    return px_acc/len(y_pred)


def mean_pixel_accuracy(y_pred, y_true):
    assert y_pred.shape==y_true.shape, "DiffDim: Different dimensions of matrices!"
        
    px_acc = 0
    for i in range(len(y_pred)):
        eval_segm = y_pred[i]
        gt_segm = y_true[i]

        cl, n_cl = extract_classes(gt_segm)
        eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        sum_n_ii = 0
        sum_t_i  = 0
        
        px_acc = 0
        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            sum_n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            sum_t_i  = np.sum(curr_gt_mask)
    
            if (sum_t_i == 0):
                px_acc += 0
            else:
                px_acc += sum_n_ii / sum_t_i
        
    mean_px_acc = px_acc / n_cl / len(y_pred)

    return mean_px_acc


def binary_IoU(y_pred, y_true):
    
    intersection = np.sum(np.abs(y_true * y_pred))
    union = np.sum(np.abs(y_true) + np.abs(y_pred)) - intersection

    if union==0:
        return 1.0
    else:
        iou = intersection / union
        return iou


def binary_Dice(y_pred, y_true):
    """ F1 Score for Binary Segmentation
    """
    intersection = np.sum(np.abs(y_true * y_pred))
    union = np.sum(np.abs(y_true) + np.abs(y_pred))

    if union==0:
        return 1.0
    else:
        dice = 2*intersection / union
        return dice



# BF Score
def calc_precision_recall(contours_a, contours_b, threshold):
    """ For precision, contours_a==GT & contours_b==Prediction
        For recall, contours_a==Prediction & contours_b==GT
    """

    x = contours_a
    y = contours_b

    xx = np.array(x)
    hits = []
    for yrec in y:
        d = np.square(xx[:,0] - yrec[0]) + np.square(xx[:,1] - yrec[1])
        hits.append(np.any(d < threshold*threshold))
    top_count = np.sum(hits)

    try:
        precision_recall = top_count / len(y)
    except ZeroDivisionError:
        precision_recall = 0

    return precision_recall, top_count, len(y)


def bfscore(y_pred, y_true, threshold=2):
    """ computes the BF (Boundary F1) contour matching score between the predicted and GT segmentation
        https://github.com/minar09/bfscore_python
    """
    assert y_pred.shape==y_true.shape,"DiffDim: Different dimensions of matrices!"
            
    bf_score = 0
    for i in range(len(y_pred)):
        
        gt = y_true[i].astype(np.uint8)    
        pr = y_pred[i].astype(np.uint8)    

        classes, n_cl = extract_classes(gt)

        # Define bfscore variable (initialized with zeros)
        bfscores = np.zeros(n_cl, dtype=float) + np.nan

        for target_class in classes: # Iterate over classes

            if target_class == 0: # Skip background
                continue

            # calculate contours of ground-truth
            gt[gt!=target_class] = 0

            contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # Find contours of the shape

            contours_gt = []
            for i in range(len(contours)):
                for j in range(len(contours[i])):
                    contours_gt.append(contours[i][j][0].tolist())
            
            # calculate contours of prediction
            pr[pr!=target_class] = 0

            contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            contours_pr = []
            for i in range(len(contours)):
                for j in range(len(contours[i])):
                    contours_pr.append(contours[i][j][0].tolist()) # Find contours of the shape

            # calculate precision & recall
            precision, numerator, denominator = calc_precision_recall(
                contours_gt, contours_pr, threshold) # Precision

            recall, numerator, denominator = calc_precision_recall(
                contours_pr, contours_gt, threshold) # Recall

            f1 = 0
            try:
                f1 = 2*recall*precision/(recall+precision) # F1 score
            except:
                f1 = np.nan

            bfscores[target_class] = f1
        
        bf_score += np.sum(bfscores[1:])/len(classes[1:])

    return bf_score/len(y_pred)