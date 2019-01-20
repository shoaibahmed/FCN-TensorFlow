import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET

dataBaseDir = '/home/pkhan/Dataset/icdar_str_devkit/data/'
IoU_THRESHOLDS = ['0.5']
COL_PRED_THRESHOLD = 0.3
ROW_PRED_THRESHOLD = 0.2
MODEL_EPOCH = 150

def bbox_intersection_over_union(boxA, boxB):
    """
    :param boxA: value of boxA in [xmin ymin xmax ymax] form
    :param boxB: value of boxB in [xmin ymin xmax ymax] form
    :return: iou of boxA and boxB
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    #interArea = (xB - xA + 1) * (yB - yA + 1)
    interArea = max(0,xB - xA + 1) * max(0,yB - yA + 1)


    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)



    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def loadGTAnnotationsFromXML(xml_path, classes):
    """
    This function extracts bounding boxes from given xml file
    :param xml_path: path of the xml file
    :param classes: class name. For this code either 'row' or 'column'
    :return: bounding boxes list
    """

    if not os.path.exists(xml_path):
        print ("Error: Unable to locate XML file %s" % (xml_path))
        exit(-1)


    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    #num_objs = len(objs)

    # Load object bounding boxes into a data frame.
    boundingBoxes = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls = obj.find('name').text.lower().strip()
        if cls in classes:
            boundingBoxes.append([x1, y1, x2, y2,  cls])

    return boundingBoxes
def loadGTAnnotationsFromXMLString(xml, classes):
    """
    This function extracts bounding boxes from given xml file
    :param xml: xml string to be parsed
    :param classes: class name. For this code either 'row' or 'column'
    :return: bounding boxes list
    """


    tree = ET.fromstring(xml)
    objs = tree.findall('object')
    #num_objs = len(objs)

    # Load object bounding boxes into a data frame.
    boundingBoxes = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        # cls = self._class_to_ind[obj.find('name').text.lower().strip()]
        cls = obj.find('name').text.lower().strip()
        if cls in classes:
            boundingBoxes.append([x1, y1, x2, y2,  cls])

    return boundingBoxes

def computeStatistics(imgName, predBoxesXmlString, statistics, classes, originalImage,cls_name):
    """
    This function computes statistics for an image by computing IOU of predicted bounding boxes and ground truth bounding boxes
    :param imgName: This is required to load original image annotations
    :param statistics: dictionary containing no. of TP, FP and FN for rows and columns
    :param classes:
    :param cls_name: string containing either 'row' or 'column'
    :return: returns new computed statistics
    """

    orgImgHeight, orgImgWidth, _ = originalImage.shape
    predBoxes = loadGTAnnotationsFromXMLString(predBoxesXmlString, classes)
    xml_path = os.path.join(dataBaseDir, 'Annotations_original', imgName[:-4] + '.xml')
    gtBBoxes = loadGTAnnotationsFromXML(xml_path, classes)
    gtBBoxesFiltered =[]
    for box in gtBBoxes:
        if box[4] == cls_name:
            gtBBoxesFiltered.append(box)

    predictListFilter = []
    for box in predBoxes:
        if box[4] == cls_name:
            predictListFilter.append(box)

    matchedGTBBox = [0] * len(gtBBoxesFiltered)

    img_path = os.path.join(dataBaseDir, 'Images_original', imgName)
    img = cv2.imread(img_path)
    height, width, channels = img.shape

    for thresh in IoU_THRESHOLDS:
        for prBBoxIdx, predBox in enumerate(predictListFilter):
            predBox_Normalized =[]
            predBox_Normalized[:] = predBox[:]
            predBox_Normalized[0] = float(predBox[0] / orgImgWidth)
            predBox_Normalized[1] = float(predBox[1] / orgImgHeight)
            predBox_Normalized[2] = float(predBox[2] / orgImgWidth)
            predBox_Normalized[3] = float(predBox[3] / orgImgHeight)
            bboxMatchedIdx = -1
            for gtBBoxIdx, gtBBox in enumerate(gtBBoxesFiltered):

                gtBBox_Normalized = []
                gtBBox_Normalized[:] = gtBBox[:]

                gtBBox_Normalized[0] = float(gtBBox[0] / width)
                gtBBox_Normalized[1] = float(gtBBox[1] / height)
                gtBBox_Normalized[2] = float(gtBBox[2] / width)
                gtBBox_Normalized[3] = float(gtBBox[3] / height)

                # Compute IoU
                iou = bbox_intersection_over_union(gtBBox_Normalized, predBox_Normalized)
                if ((iou > float(thresh)) and (gtBBox[4] == predBox[4] and cls_name == predBox[4])):
                    if not matchedGTBBox[gtBBoxIdx]:
                        bboxMatchedIdx = gtBBoxIdx
                        break

            if (bboxMatchedIdx != -1):
                statistics[cls_name][thresh]["truePositives"] += 1
                matchedGTBBox[bboxMatchedIdx] = 1
            else:
                statistics[cls_name][thresh]["falsePositives"] += 1

        # All the unmatched bboxes are false negatives
        for idx, gtBBox in enumerate(gtBBoxesFiltered):
            if not matchedGTBBox[idx] and gtBBox[4] == cls_name:
                statistics[cls_name][thresh]["falseNegatives"] += 1

    return statistics

def getBoundingBoxes(imgName, rows, cols,predictedRowSegMask, predictedColSegMask):
    xminColsList = []
    yminColsList = []
    widthColsList = []
    heightColsList = []

    rows = rows.sort_values(['y1'], ascending=[True])
    cols = cols.sort_values(['x1'], ascending=[True])

    rowCount = len(rows)
    colCount = len(cols)

    height = 0
    if rowCount > 0:
        height = rows.iloc[rowCount - 1]['y1'] - rows.iloc[0]['y1']

    # Create Bounding Boxes for Columns
    for col in range(colCount):
        width = 0
        if col < colCount - 1:
            xmin = cols.iloc[col]['x1']
            ymin = rows.iloc[0]['y1']

            width = cols.iloc[col + 1]['x1'] - cols.iloc[col]['x1']

        if height > 0 and width > 0:
            xmax = xmin + width
            ymax = ymin + height
            xminColsList.append(xmin)
            yminColsList.append(ymin)
            widthColsList.append(xmax)
            heightColsList.append(ymax)

    # Create Bounding Boxes for Rows

    xminRowsList = []
    yminRowsList = []
    widthRowsList = []
    heightRowsList = []

    if colCount > 0:
        width = cols.iloc[colCount - 1]['x1'] - cols.iloc[0]['x1']

    for row in range(rowCount):
        height = 0
        if row < rowCount - 1:
            xmin = cols.iloc[0]['x1']
            ymin = rows.iloc[row]['y1']

            height = rows.iloc[row + 1]['y1'] - rows.iloc[row]['y1']

            if height > 0 and width > 0:
                xmax = xmin + width
                ymax = ymin + height
                xminRowsList.append(xmin)
                yminRowsList.append(ymin)
                widthRowsList.append(xmax)
                heightRowsList.append(ymax)

    colXml = ''
    rowXml = ''

    for (xmin, ymin, xmax, ymax) in zip(xminColsList, yminColsList, widthColsList,
                                        heightColsList):
        colMask = predictedColSegMask[0, :, :, 0]
        noOfColPredictions = np.sum((colMask[ymin:ymax, xmin:xmax] == 1))
        totalColPredRegion = (xmax - xmin) * (ymax - ymin)
        if (noOfColPredictions / totalColPredRegion) >= COL_PRED_THRESHOLD:
            newItem = '<object>\n' + '<name>column</name>\n' + '<bndbox>' + '<xmin>' + str(xmin) + '</xmin>' + '<ymin>' + str(ymin) + '</ymin>' + '<xmax>' + str(xmax)  + '</xmax>' + '<ymax>' + str(ymax ) + '</ymax>' + '</bndbox>' + '\n' + '</object>\n'

            if newItem not in colXml:
                colXml = colXml + newItem

    for (xmin, ymin, xmax, ymax) in zip(xminRowsList, yminRowsList, widthRowsList, heightRowsList):
        rowMask = predictedRowSegMask[0, :, :, 0]

        noOfRowPredictions = np.sum((rowMask[ymin:ymax, xmin:xmax] == 1))
        totalRowPredRegion = (xmax - xmin) * (ymax - ymin)

        if (noOfRowPredictions / totalRowPredRegion) >= ROW_PRED_THRESHOLD:
            newItem = '<object>\n' + '<name>row</name>\n' + '<bndbox>' + '<xmin>' + str(
                    xmin) + '</xmin>' + '<ymin>' + str(ymin) + '</ymin>' + '<xmax>' + str(
                    xmax) + '</xmax>' + '<ymax>' + str(ymax) + '</ymax>' + '</bndbox>' + '\n' + '</object>\n'
            if newItem not in rowXml:
                rowXml = rowXml + newItem

    boundingBoxXml = '<?xml version="1.0" encoding="utf-8"?> \n' + '<annotation>\n' + colXml + rowXml + '</annotation>\n'

    return boundingBoxXml

def saveStatistics(statistics, path):
    """
    This function saved computed statistics at specified path
    :param statistics: computed statistics to be saved
    :param path: Path at which statistics must be saved
    :return:
    """

    EXPERIMENT_NAME = "trainer_fcn_str_" + str(MODEL_EPOCH) + "ep"
    outputFile = open(os.path.join(path, 'output-stats-' + EXPERIMENT_NAME + '.txt'), 'w')
    for cls in statistics.keys():
        for thresh in statistics[cls].keys():
            if (statistics[cls][thresh]["truePositives"] == 0) and (statistics[cls][thresh]["falsePositives"] == 0):
                precision = 1.0
            else:
                precision = float(statistics[cls][thresh]["truePositives"]) / float(
                    statistics[cls][thresh]["truePositives"] + statistics[cls][thresh]["falsePositives"])
            if (statistics[cls][thresh]["truePositives"] == 0) and (statistics[cls][thresh]["falseNegatives"] == 0):
                recall = 1.0
            else:
                recall = float(statistics[cls][thresh]["truePositives"]) / float(
                    statistics[cls][thresh]["truePositives"] + statistics[cls][thresh]["falseNegatives"])
            if (precision == 0.0) and (recall == 0.0):
                fMeasure = 0.0
            else:
                fMeasure = 2 * ((precision * recall) / (precision + recall))

            statistics[cls][thresh]["precision"] = precision
            statistics[cls][thresh]["recall"] = recall
            statistics[cls][thresh]["fMeasure"] = fMeasure

            print("--------------------------------")
            print("Class: %s" % (cls))
            print("IoU Threshold: %s" % (thresh))
            print("True Positives: %d" % (statistics[cls][thresh]["truePositives"]))
            print("False Positives: %d" % (statistics[cls][thresh]["falsePositives"]))
            print("False Negatives: %d" % (statistics[cls][thresh]["falseNegatives"]))
            print("Precision: %f" % (precision))
            print("Recall: %f" % (recall))
            print("F-Measure: %f" % (fMeasure))

            outputFile.write("Class: %s" % (cls) + "\n")
            outputFile.write("IoU Threshold: %s" % (thresh) + "\n")
            outputFile.write("True Positives: %d" % (statistics[cls][thresh]["truePositives"]) + "\n")
            outputFile.write("False Positives: %d" % (statistics[cls][thresh]["falsePositives"]) + "\n")
            outputFile.write("False Negatives: %d" % (statistics[cls][thresh]["falseNegatives"]) + "\n")
            outputFile.write("Precision: %f" % (precision) + "\n")
            outputFile.write("Recall: %f" % (recall) + "\n")
            outputFile.write("F-Measure: %f" % (fMeasure) + "\n")
            outputFile.write("--------------------------------\n")

    outputFile.close()

def drawBoundingBoxes(img,fileName,directory, xml, classes):

    """
    This function draws both predicated and ground truth bounding boxes
    :param img:
    :param fileName: Name of the original image. This is used to get original image dimensions as well load ground truth annotations
    :param directory: Location of the file system where image will be saved after drawing bounding boxes
    :param xml: predicated bounding box xml
    :param classes: List of classes. In this case 'row' and 'column'
    """
    # Draw predicted bounding boxes
    _,height, width, channels = img.shape
    if img is not None:
        rowImg = img[0].copy()
        colImg = img[0].copy()

    fileNameRoot, fileNameExt = os.path.splitext(fileName)
    originalImagePath= os.path.join(dataBaseDir,'Images_original',fileName)
    orgHeight, orgWidth, _ = cv2.imread(originalImagePath).shape

    tree = ET.fromstring(xml)
    objs = tree.findall('object')

    # Load object bounding boxes into a data frame.
    boundingBoxes = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')

        cls = obj.find('name').text.lower().strip()
        rowBoundingBoxes = []
        colBoundingBoxes = []
        if(cls == classes[0]):
            rowBoundingBoxes.append(bbox)
        elif(cls == classes[1]):
            colBoundingBoxes.append(bbox)

        for bbox in rowBoundingBoxes:

            # Make pixel indexes 0-based
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            cv2.rectangle(rowImg, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for bbox in colBoundingBoxes:
            # Make pixel indexes 0-based
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            cv2.rectangle(colImg, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw Ground Truth Bounding Boxes
        gtXml = os.path.join(dataBaseDir,'Annotations_original', fileNameRoot +'.xml')

        tree = ET.parse(gtXml)
        objs = tree.findall('object')

        # Load object bounding boxes into a data frame.
        boundingBoxes = []
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')

            cls = obj.find('name').text.lower().strip()
            rowBoundingBoxes = []
            colBoundingBoxes = []
            if (cls == classes[0]):
                rowBoundingBoxes.append(bbox)
            elif (cls == classes[1]):
                colBoundingBoxes.append(bbox)

            for bbox in rowBoundingBoxes:

                # Make pixel indexes 0-based
                x1 = int(bbox.find('xmin').text) - 1
                y1 = int(bbox.find('ymin').text) - 1
                x2 = int(bbox.find('xmax').text) - 1
                y2 = int(bbox.find('ymax').text) - 1
                x1 = float((x1 / orgWidth))
                x1 = round(x1 * width)
                y1 = float((y1 / orgHeight))
                y1 = round(y1 * height)
                x2 = float((x2 / orgWidth))
                x2 = round(x2 * width)
                y2 = float((y2 / orgHeight))
                y2 = round(y2 * height)
                cv2.rectangle(rowImg, (x1, y1), (x2, y2), (0, 255, 0), 2)

            rowOutputFileName = os.path.join(directory, fileNameRoot + "-" + classes[0] + '-BBox' + fileNameExt)
            cv2.imwrite(rowOutputFileName, rowImg)
            colOutputFileName = os.path.join(directory, fileNameRoot + "-" + classes[1] + '-BBox' + fileNameExt)

            for bbox in colBoundingBoxes:
                # Make pixel indexes 0-based
                x1 = int(bbox.find('xmin').text) - 1
                y1 = int(bbox.find('ymin').text) - 1
                x2 = int(bbox.find('xmax').text) - 1
                y2 = int(bbox.find('ymax').text) - 1
                x1 = float((x1 / orgWidth))
                x1 = round(x1 * width)
                y1 = float((y1 / orgHeight))
                y1 = round(y1 * height)
                x2 = float((x2 / orgWidth))
                x2 = round(x2 * width)
                y2 = float((y2 / orgHeight))
                y2 = round(y2 * height)
                cv2.rectangle(colImg, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(colOutputFileName, colImg)
