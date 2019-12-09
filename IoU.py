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

_path = os.getcwd()

os.chdir(args.detFolder)
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
IoU_count = [0]*10
his = []
for f in files:
    nameOfImage = f.replace(".txt", "")
    gt_path = os.path.join(args.gtFolder,f)
    det_path = os.path.join(args.detFolder,f)
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

        IoU_max = 0
        for line2 in fh_all_splitLine:

            x = float(line2[0])
            y = float(line2[1])
            x2 = float(line2[2])
            y2 = float(line2[3])

            x_max = max(x,gt_x)
            y_max = max(y,gt_y)
            x2_min = min(x2,gt_x2)
            y2_min = min(y2,gt_y2)

            if x2_min > x_max and y2_min > y_max:
                TP = ((x2_min-x_max)*(y2_min-y_max))
                AoU = ((gt_x2-gt_x)*(gt_y2-gt_y))+((x2-x)*(y2-y))-TP
                # print('TP',TP)
                # print('AoU',AoU)
                IoU = TP/AoU
                # print('IoU',IoU)
                IoU_max = max(IoU,IoU_max)

        his.append(IoU_max)
        id = int(IoU_max * 10.0)
        IoU_count[id] += 1.0

    gt1.close()

print('gt_BBox:',gt_bbox_count)
print(IoU_count)
# print(max(his))
# l1 = [i/gt_bbox_count for i in IoU_count]
# print('l1',l1)
# x = [j*0.1 for j in range(10)]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(his, normed = True)
ax.set_xlabel('IoU')
ax.set_ylabel('( num of bbox / all bbox ) * 10')
fig.savefig('./output_hist.png')
