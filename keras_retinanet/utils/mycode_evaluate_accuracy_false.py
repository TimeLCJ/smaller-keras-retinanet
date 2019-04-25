"""
主要实现一个评估准确率和误检率的模块
"""
from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import numpy as np
import os

import cv2


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations


def evaluate_accuracy_false_rate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names' label to accuracy rate and A dict mapping class names' label to false rate.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)

    # 创建空的准确率和误检率的dict
    accuracy_rates = {}
    false_rates = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        non_target_recognize_as_target = np.zeros((0,))
        correctly_identify_target  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                if d[4] >= 0.5: # 置信度大于0.5才会把这个预测框计算在内
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        non_target_recognize_as_target = np.append(non_target_recognize_as_target, 1)
                        correctly_identify_target  = np.append(correctly_identify_target, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        non_target_recognize_as_target = np.append(non_target_recognize_as_target, 0)
                        correctly_identify_target  = np.append(correctly_identify_target, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        non_target_recognize_as_target = np.append(non_target_recognize_as_target, 1)
                        correctly_identify_target  = np.append(correctly_identify_target, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            accuracy_rates[label] = 0
            false_rates[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        non_target_recognize_as_target = non_target_recognize_as_target[indices]
        correctly_identify_target  = correctly_identify_target[indices]

        # compute false positives and true positives
        non_target_recognize_as_target = np.cumsum(non_target_recognize_as_target)
        correctly_identify_target  = np.cumsum(correctly_identify_target)

        # 计算准确率和误检率
        if np.shape(correctly_identify_target) == (0,): # 防止正确识别的该种害虫数量为0,而抛出错误
            accuracy_rate = 0
        else:
            accuracy_rate    = np.max(correctly_identify_target / num_annotations)
        if np.shape(non_target_recognize_as_target) == (0,): # 防止非目标害虫识别为目标害虫的数量为0，而抛出错误
            false_rate = 0
        else:
            false_rate = np.max(non_target_recognize_as_target / np.maximum(correctly_identify_target + non_target_recognize_as_target, np.finfo(np.float64).eps))

        # compute average precision
        accuracy_rates[label] = accuracy_rate
        false_rates[label] = false_rate

    return accuracy_rates, false_rates