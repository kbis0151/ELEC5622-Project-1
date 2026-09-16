from os import listdir, rename
from os.path import isfile, join
import random

mypath = "./"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nii")]
random.shuffle(onlyfiles)
newnames = []
for i in range(50):
	newnames.append("Data_{0:02d}.nii".format(i))
with open("data_labels.csv", "w+") as f:
	for i, j in zip(onlyfiles, newnames):
		f.write(i + ", " + j + ", " + i[0:2] + "\n")
		rename(i, j)
