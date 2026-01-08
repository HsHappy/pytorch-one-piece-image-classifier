def moveRandomImages(sourcePath, destinationPath, fileToPickCount, keepOriginalName = True):
    from glob import glob
    from matplotlib.pylab import imread, imsave
    from random import randint as r
    from os import remove
    from pathlib import Path

    Path(destinationPath).mkdir(parents = True, exist_ok = True)

    ogFiles = glob((sourcePath+"\\*"))
    ogFilesCount = len(ogFiles)
    #print("count: ", ogFilesCount)

    pickedFileIndexes = []
    for count in range(fileToPickCount):
        index = r(0, (ogFilesCount - 1))
        while (index in pickedFileIndexes):
            index = r(0, (ogFilesCount - 1))
        #print(index)
        pickedFileIndexes.append(index)

    newIndex = 0
    for index in pickedFileIndexes:
        #print("-", index)
        currentFile = imread(ogFiles[index])
        #print(currentFile, type(currentFile))
        fileName = ogFiles[index].split('\\')[-1].split('.')[0]
        if keepOriginalName:
            imsave((f"{destinationPath}\\Randomly-Moved-Image{newIndex}({fileName}).png"), currentFile)
        else:
            imsave((f"{destinationPath}\\Randomly-Moved-Image{newIndex}.png"), currentFile)
        remove(ogFiles[index])
        newIndex += 1


def copyRandomImages(sourcePath, destinationPath, fileToPickCount, keepOriginalName = True):
    from glob import glob
    from matplotlib.pylab import imread, imsave
    from random import randint as r
    from pathlib import Path

    Path(destinationPath).mkdir(parents = True, exist_ok = True)
    
    ogFiles = glob((sourcePath+"\\*"))
    ogFilesCount = len(ogFiles)
    #print("count: ", ogFilesCount)

    pickedFileIndexes = []
    for count in range(fileToPickCount):
        index = r(0, (ogFilesCount - 1))
        while (index in pickedFileIndexes):
            index = r(0, (ogFilesCount - 1))
        #print(index)
        pickedFileIndexes.append(index)

    newIndex = 0
    for index in pickedFileIndexes:
        #print("-", index)
        currentFile = imread(ogFiles[index])
        #print(currentFile, type(currentFile))
        fileName = ogFiles[index].split('\\')[-1].split('.')[0]
        if keepOriginalName:
            imsave((f"{destinationPath}\\Randomly-Copied-Image{newIndex}({fileName}).png"), currentFile)
        else:
            imsave((f"{destinationPath}\\Randomly-Copied-Image{newIndex}.png"), currentFile)
        newIndex += 1


def moveAllImages(sourcePath, destinationPath, keepOriginalName = True):
    from glob import glob
    from matplotlib.pylab import imread, imsave
    #from random import randint as r
    from os import remove
    from pathlib import Path

    Path(destinationPath).mkdir(parents = True, exist_ok = True)
    
    ogFiles = glob((sourcePath+"\\*"))
    #ogFilesCount = len(ogFiles)
    #print("count: ", ogFilesCount)

    fileIndex = 0
    for file in ogFiles:
        #print("-", index)
        currentFile = imread(file)
        fileName = file.split('\\')[-1].split('.')[0]
        if keepOriginalName:
            imsave((f"{destinationPath}\\Moved-Image{fileIndex}({fileName}).png"), currentFile)
        else:
            imsave((f"{destinationPath}\\Moved-Image{fileIndex}.png"), currentFile)
        remove(file)
        fileIndex += 1


def copyAllImages(sourcePath, destinationPath, keepOriginalName = True):
    from glob import glob
    from matplotlib.pylab import imread, imsave
    from pathlib import Path

    Path(destinationPath).mkdir(parents = True, exist_ok = True)
    
    ogFiles = glob((sourcePath+"\\*"))
    #ogFilesCount = len(ogFiles)
    #print("count: ", ogFilesCount)

    fileIndex = 0
    for file in ogFiles:
        #print("-", file)
        currentFile = imread(file)
        fileName = file.split('\\')[-1].split('.')[0]
        if keepOriginalName:
            imsave((f"{destinationPath}\\Copied-Image{fileIndex}({fileName}).png"), currentFile)
        else:
            imsave((f"{destinationPath}\\Copied-Image{fileIndex}.png"), currentFile)
        fileIndex += 1

    

from pathlib import Path
from glob import glob

###
FULL_DATASET_DIRECTORY = "C:\\Users\\Taha\\Desktop\\okul\\veri bilimi 3.1\\final projes\\main\\FULL_DATASET\\Data\\Data"
TARGET_DATASET_DIRECTORY = "C:\\Users\\Taha\\Desktop\\okul\\veri bilimi 3.1\\final projes\\main\\FULL_DATASET_FIXED_NAMES\\"
###

mainDirectory = str(Path(__file__).parent.parent)
#print(mainDirectory)

#folders = glob((mainDirectory + "\\FULL_DATASET\\Data\\Data\\*"))
folders = glob(FULL_DATASET_DIRECTORY + "\\*")

for folder in folders:
    className = folder.split('\\')[-1]
    print(className)
    copyAllImages(folder, f"{TARGET_DATASET_DIRECTORY}\\{className}", False)