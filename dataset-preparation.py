import cv2
import numpy as np
import os
import math


def blurImage(image):
    bluredImage = cv2.blur(image, (5, 5))
    return bluredImage

def noiseWithDCT(image, dctSize = [25, 25]):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clipDCT = cv2.dct(np.float32(grayImage))[:dctSize[0], :dctSize[1]]
    reconstructedClip = cv2.idct(clipDCT)
    resizedImage = cv2.resize(reconstructedClip, (256, 256))
    RGBImage = cv2.cvtColor(resizedImage, cv2.COLOR_GRAY2RGB)
    return RGBImage

def pix2pixDataset(datasetPath, trainSetFolderName,testSetFolderName = None,trainSetCount = None ,testSetCount = None, defromationFunction = blurImage, outputFolder = 'blur-dataset-'):
    counter = 1
    for filename in os.listdir(f"{datasetPath}\\{trainSetFolderName}"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            imagePath = f'{datasetPath}\\{trainSetFolderName}\\{filename}'
            rawImage = cv2.imread(imagePath)
            bluredImage = defromationFunction(rawImage)
            imagePair = np.concatenate((rawImage, bluredImage), axis=1)
            cv2.imwrite(f'{datasetPath}\\{outputFolder}-train\\{counter}.jpg', imagePair)
            counter += 1

    counter = 1
    if testSetFolderName != None:
        for filename in os.listdir(f"{datasetPath}\\{testSetFolderName}"):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                imagePath = f'{datasetPath}\\{testSetFolderName}\\{filename}'
                rawImage = cv2.imread(imagePath)
                bluredImage = defromationFunction(rawImage)
                imagePair = np.concatenate((rawImage, bluredImage), axis=1)
                cv2.imwrite(f'{datasetPath}\\{outputFolder}-test\\{counter}.png', imagePair)
                counter += 1


def divideDataset(datasetPath, size):
    os.mkdir(f"{datasetPath}\\train")
    os.mkdir(f"{datasetPath}\\test")
    trainSize = math.floor(size * 0.7)
    # testSize = math.ceil(size * 0.3)

    trainCounter = 0
    testCounter = 0
    for filename in os.listdir(f"{datasetPath}"):
        if trainCounter < trainSize:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                trainCounter += 1
                os.rename(f"{datasetPath}\\{filename}", f"{datasetPath}\\train\\{trainCounter}.jpg")

        else:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                testCounter += 1
                os.rename(f"{datasetPath}\\{filename}", f"{datasetPath}\\test\\{testCounter}.jpg")

def main():
    generateBluredFacade = True
    generateDCTLayoutClips = False
    generateDCTFacadeDataset = False
    divideDatasetTrainTest = False

    # Generate the facade data set to be operated with pix2pix
    if generateBluredFacade:
        pix2pixDataset('C:\\Users\\adelmahm\\Desktop\Masters\\Facade-dataset', 'trainA', 'testA', 400, 106, blurImage, outputFolder= 'blur-dataset-5x5')


    if generateDCTLayoutClips:
        pix2pixDataset('C:\\Users\\adelmahm\\Desktop\Masters\\layout-to-image', 'clips', trainSetCount = 400, defromationFunction = noiseWithDCT, outputFolder='DCT-deformation-dataset')

    if generateDCTFacadeDataset:
        pix2pixDataset('C:\\Users\\adelmahm\\Desktop\Masters\\Facade-dataset', 'trainA', 'testA',  400, 106, defromationFunction = noiseWithDCT, outputFolder='DCT-deformation-Facade-dataset')

    if divideDatasetTrainTest:
        divideDataset('C:\\Users\\adelmahm\\Desktop\Masters\\layout-to-image\\DCT-deformation-dataset-train', 3055)

if __name__ == "__main__":
    main()
