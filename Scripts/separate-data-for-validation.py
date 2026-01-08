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
        fileName = ogFiles[index].split('\\')[-1].split('.')[0]
        if keepOriginalName:
            imsave((f"{destinationPath}\\Randomly-Copied-Image{newIndex}({fileName}).png"), currentFile)
        else:
            imsave((f"{destinationPath}\\Randomly-Copied-Image{newIndex}.png"), currentFile)
        newIndex += 1


def moveAllImages(sourcePath, destinationPath, keepOriginalName = True):
    from glob import glob
    from matplotlib.pylab import imread, imsave
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
        #print("-", index)
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
FULL_DATASET_DIRECTORY = "C:\\" #Enter dataset path here
TARGET_DATASET_DIRECTORY = "C:\\" #Enter target path here
DATASET_VALIDATE_TO_TOTAL_RATIO = 0.2 #Enter the desired ratio of validation/full data here
###


mainDirectory = str(Path(__file__).parent.parent)

folders = glob(FULL_DATASET_DIRECTORY + "\\*")


for folder in folders:
    className = folder.split('\\')[-1]
    print(className)
    trainDirectory = f"{TARGET_DATASET_DIRECTORY}\\train\\{className}"
    validateDirectory = f"{TARGET_DATASET_DIRECTORY}\\validate\\{className}"
    copyAllImages(folder, trainDirectory)

    currentFolderImageCount = len(glob(folder + "\\*"))
    imageToValidateCount = int(currentFolderImageCount * DATASET_VALIDATE_TO_TOTAL_RATIO)
    
    print("currentFolderImageCount:", currentFolderImageCount, "imageToValidateCount:", imageToValidateCount)
    moveRandomImages(trainDirectory, validateDirectory, imageToValidateCount)






