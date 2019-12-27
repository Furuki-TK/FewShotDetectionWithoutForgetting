import os
import argparse
import glob
import os
from matplotlib import pyplot as plt

currentPath = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument(
    '-gt',
    '--gtfolder',
    dest='gtFolder',
    default=os.path.join(currentPath, 'groundtruths'),
    metavar='',
    help='folder containing your ground truth bounding boxes')

parser.add_argument(
    '-det',
    '--detfolder',
    dest='detFolder',
    default=os.path.join(currentPath, 'detections'),
    metavar='',
    help='folder containing your detected bounding boxes')

args = parser.parse_args()


class_names = ['aeroplane', 'bicycle','bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog' ,'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

IoU = {}
Rel = {}
for class_name in class_names:
    IoU[class_name] = []
    Rel[class_name] = []

_path = os.getcwd()

# derfolder = args.detFolder + str(args.ch)
# ch_name = derfolder.split("/")[-1].split("_")[-1]
derfolder = args.detFolder
os.chdir(derfolder)
files = glob.glob("*.txt")
files.sort()
os.chdir(_path)

# Read GT detections from txt file
# Each line of the files in the groundtruths folder represents a ground truth bounding box
# (bounding boxes that a detector should detect)
# Each value of each line is  "class_id, x, y, width, height" respectively
# Class_id represents the class of the bounding box
# x, y represents the most top-left coordinates of the bounding box
# x2, y2 represents the most bottom-right coordinates of the bounding box

gt_bbox_count = 0

for f in files:
    nameOfImage = f.replace(".txt", "")
    gt_path = os.path.join(args.gtFolder,f)
    det_path = os.path.join(derfolder,f)
    gt1 = open(gt_path, "r")
    fh1 = open(det_path, "r")
    fh_all_splitLine = []
    for line_fh1 in fh1:
        line_fh1 = line_fh1.replace("\n", "")
        if line_fh1.replace(' ', '') == '':
            continue
        fh_splitLine = line_fh1.split(" ")
        fh_all_splitLine.append(fh_splitLine)

    fh1.close()

    for line in gt1:
        gt_bbox_count += 1
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        gt_splitLine = line.split(" ")
        idClass = (gt_splitLine[0])
        gt_x = float(gt_splitLine[1])
        gt_y = float(gt_splitLine[2])
        gt_x2 = float(gt_splitLine[3])
        gt_y2 = float(gt_splitLine[4])

        IoU_val = 0.0
        for line2 in fh_all_splitLine:

            det_class = (line2[0])
            det_value = float(line2[1])
            x = float(line2[2])
            y = float(line2[3])
            x2 = float(line2[4])
            y2 = float(line2[5])

            x_max = max(x,gt_x)
            y_max = max(y,gt_y)
            x2_min = min(x2,gt_x2)
            y2_min = min(y2,gt_y2)

            if x2_min > x_max and y2_min > y_max and det_class == idClass:
                TP = ((x2_min-x_max)*(y2_min-y_max))
                AoU = ((gt_x2-gt_x)*(gt_y2-gt_y))+((x2-x)*(y2-y))-TP
                # print('TP',TP)
                # print('AoU',AoU)
                IoU_val = TP/AoU
                # print('IoU',IoU)
            else:
                IoU_val = 0.0

            IoU[det_class].append(IoU_val)
            Rel[det_class].append(det_value)

    gt1.close()

print('gt_BBox:',gt_bbox_count)
print('det_BBox via class')
for c, v in IoU.items():
    print('  {0}:{1}'.format(c,len(v)))


for class_name in class_names:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(IoU[class_name], Rel[class_name])
    ax.set_xlabel('IoU')
    ax.set_ylabel('Reliability')

    fig.savefig('./results/plt_IoU_Rel_'+class_name+'.png')
    plt.close()
