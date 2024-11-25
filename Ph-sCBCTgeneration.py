# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:22:33 2024
Tested on python version 3.9
To install the required packages with pip 
set PYTHONUTF8=1
pip install medpy numpy opencv-python-headless SimpleITK dicom-mask scipy tqdm pillow pandas matplotlib

@author: RTresearchPC
"""

from medpy.io import load
import numpy as np
import cv2
import SimpleITK as sitk
from dicom_mask.convert import struct_to_mask
import os
from medpy.io import load,save
import numpy as np
import SimpleITK as sitk
import cv2
from scipy import ndimage
from tqdm import tqdm
import random
from scipy.ndimage import zoom
from scipy import signal
from PIL import ImageEnhance
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
 
#%%

def Mask_CT(CT, kernelsize):
    maskbool = np.copy(CT)
    maskbool = maskbool > 100
    mask=maskbool.astype(np.uint8)

    # mask: closure
    kernel=np.ones((kernelsize,kernelsize), np.uint16)      #made it smaller
    mask = cv2.erode(mask,kernel)
    
    return mask,maskbool


def Preprocess_WaterPhantom(InputFolder, LN_images, LN_start, LN_stop, OutputFolder,OutputName):
    """ 
    Preprocess a set of water phantom DICOM images for further analysis.

    Parameters:
    InputFolder: str
        Folder containing the raw water phantom images in DICOM format.
    LN_images: list
        An array of the names of the water phantom scans. The order determines how the images are concatenated.
    LN_start: list
        An array specifying the first slice to include in the processed phantom for each scan.
    LN_stop: list
        An array specifying the last slice to include in the processed phantom for each scan.
    OutputFolder: str
        Path to output folder, e.g., 'c:/output'
    OutputName: str
        Output filename processed water phantom scan, e.g., 'ProcessedPhantom.mha'.
    """

    # Check that the number of images matches the provided start and stop indices
    if len(LN_images) != len(LN_start):
        raise ValueError("Number of images in LN_images is not equal to the inputs in LN_start.")
    if len(LN_images) != len(LN_stop):
        raise ValueError("Number of images in LN_images is not equal to the inputs in LN_stop.")

    # Create output folder if it does not exist
    try:
        os.makedirs(OutputFolder)
    except FileExistsError:
        pass

    TotalPhantom = []

    for i in range(len(LN_images)):  # Iterate over the list of images
        # Load the current phantom scan
        partPhantom, Header = load(os.path.join(InputFolder, LN_images[i]))

        # Crop slices based on the start and stop indices with added margins
        crop = partPhantom[:, :, LN_start[i]:LN_stop[i]] + 1024

        # Initialize TotalPhantom with the correct dimensions if it's the first iteration
        if i == 0:
            TotalPhantom = np.zeros(partPhantom.shape)[:, :, 0:1]

        # Generate a mask for the current crop and apply it
        mask, maskbool = Mask_CT(crop,25)
        maskedPhantom = crop * mask

        # Adjust intensity based on the last slice of the previous scan
        if i == 0:
            data = maskedPhantom.astype(np.float64)
            data[data == 0] = np.nan
            meanImage = np.nanmean(data[:, :, :])
            previousLast = np.nanmean(data[:, :, data.shape[2] - 1])

        elif i > 0:
            data = maskedPhantom.astype(np.float64)
            data[data == 0] = np.nan
            meanFirstSlice = np.nanmean(data[:, :, 0])
            maskedPhantom = maskedPhantom + (previousLast - meanFirstSlice)
            maskedPhantom = maskedPhantom * mask

            # Update the last slice mean for the current phantom
            data = maskedPhantom.astype(np.float64)
            data[data == 0] = np.nan
            previousLast = np.nanmean(data[:, :, data.shape[2] - 1])

        # Concatenate the processed slices to the total phantom
        TotalPhantom = np.concatenate([TotalPhantom, maskedPhantom], axis=2)

    # Remove the initial placeholder slice
    TotalPhantom = TotalPhantom[:, :, 1:TotalPhantom.shape[2]]

    # Reorder dimensions for further processing
    test = np.transpose(TotalPhantom, (2, 0, 1))

    # Identify and remove empty slices (all-zero rows or columns)
    result = (test == 0)
    sec = np.where(result.all(1))[1]
    third = np.where(result.all(2))[1]
    maskedPhantom = np.delete(np.delete(test, sec, 1), third, 2)

    # Adjust intensity values and remove noise
    data = maskedPhantom.astype(np.float64)
    data[data == 0] = np.nan
    means = np.nanmean(data[:, 1:])
    maskedPhantom = maskedPhantom - means
    maskedPhantom[maskedPhantom < -600] = 0

    # Convert the array to an image and save it
    maskedPhantom = sitk.GetImageFromArray(maskedPhantom, isVector=False)

    # Save the processed water phantom scan
    sitk.WriteImage(maskedPhantom, OutputFolder+"/"+OutputName, False)
    

#%%
def CT2Ph_sCBCT(Waterphantom_Path, pCT_Path, ContourPath, CTV_StructureName, StructuresSegmented, Factor, Output_Path, StructureNumbers):
    """
    Combines a water phantom with a pCT and creates synthetic CBCT (sCBCT) images.

    Parameters:
    Waterphantom_Path: str
        Path to the water phantom image.
    pCT_Path: str
        Path to the planning CT (pCT) image.
    ContourPath: str
        Directory containing contour files for the pCT.
    CTV_StructureName: str
        Name of the Contoured Target Volume (CTV) structure.
    StructuresSegmented: list
        List of segmented structures to create masks for.
    Factor: float
        Scaling factor applied to the phantom intensity when merging with the pCT.
    Output_Path: str
        Path to save the resulting synthetic CBCT and masks.
    StructureNumbers: list
        List of numbers assigned to each structure in the output mask.
    """
    # Load water phantom and pCT images
    ringPhantom, Header = load(Waterphantom_Path)
    pCT, Header = load(pCT_Path)

    # Generate mask for the CTV structure
    mask_CTV = struct_to_mask(ContourPath, os.listdir(ContourPath), CTV_StructureName).astype(np.uint8)
    if np.max(mask_CTV) == np.min(mask_CTV):
        print("\n No ",CTV_StructureName," P found")
    else:
        # Adjust orientation of the CTV mask to match the pCT
        mask_CTV = np.transpose(mask_CTV, (2, 1, 0))
        mask_CTV = np.flip(mask_CTV, axis=2)

        # Create a copy of the water phantom for processing
        Phantom = np.copy(ringPhantom)

        # Apply a random rotation to the phantom
        rot = random.randint(-90, 90)

        # Ensure the phantom size matches the pCT dimensions
        if Phantom.shape[2] >= pCT.shape[2] - 20:
            a = random.randint(0, 20)
            b = 20 - a
            Phantom = Phantom[:, :, a:pCT.shape[2] - b]

        # Determine the center of mass for the CTV
        center = ndimage.center_of_mass(mask_CTV)
        [x_pCT, y_pCT, z_pCT] = [
            np.round(center[0] + random.randint(-15, 15)).astype(np.int16),
            np.round(center[1] + random.randint(-15, 15)).astype(np.int16),
            np.round(center[2] + random.randint(0, 20)).astype(np.int16),
        ]
        [x_Phantom, y_Phantom, z_Phantom] = [
            np.round(Phantom.shape[0] / 2).astype(np.int16),
            np.round(Phantom.shape[1] / 2).astype(np.int16),
            np.round(Phantom.shape[2] / 2).astype(np.int16),
        ]

        # Ensure the phantom fits within the pCT volume
        if pCT.shape[0] <= (x_pCT + x_Phantom):
            x_pCT = pCT.shape[0] - x_Phantom - 2
        if 0 >= x_pCT - x_Phantom:
            x_pCT = x_Phantom + 1
        if pCT.shape[1] <= (y_pCT + y_Phantom):
            y_pCT = pCT.shape[1] - y_Phantom - 2
        if 0 >= y_pCT - y_Phantom:
            y_pCT = y_Phantom + 1
        if pCT.shape[2] <= (z_pCT + z_Phantom):
            z_pCT = pCT.shape[2] - z_Phantom - 2
        if 0 >= z_pCT - z_Phantom:
            z_pCT = z_Phantom + 1

        # Smooth the pCT values to handle intensity ranges
        ring_pCT = pCT.copy()
        ring_pCT[ring_pCT < -1024] = -1024

        # Merge the phantom into the pCT
        for x in range(Phantom.shape[0]):
            for y in range(Phantom.shape[1]):
                for z in range(Phantom.shape[2]):
                    ring_pCT[
                        (x + x_pCT - x_Phantom), 
                        (y + y_pCT - y_Phantom), 
                        (z + z_pCT - z_Phantom)
                    ] = np.round(
                        pCT[(x + x_pCT - x_Phantom), (y + y_pCT - y_Phantom), (z + z_pCT - z_Phantom)]
                        + Factor * Phantom[x, y, z]
                    ).astype(np.int16())

        # Save the synthetic CBCT image
        try:
            os.makedirs(Output_Path)
        except FileExistsError:
            pass
        save(ring_pCT, Output_Path+"\Ph-sCBCT.mha", Header)

        # Generate structure masks for segmented structures
        for i in range(len(StructuresSegmented)):
            try:
                StructMask = struct_to_mask(ContourPath, os.listdir(ContourPath), StructuresSegmented[i])
            except:
                StructMask = struct_to_mask(ContourPath, os.listdir(ContourPath), CTV_StructureName)
                print("\n No ", StructuresSegmented[i])

            if np.max(StructMask) == np.min(StructMask):
                print("\n No ", StructuresSegmented[i])

            # Adjust orientation and flip the mask to match the pCT
            StructMask = np.transpose(StructMask, (2, 1, 0))
            StructMask = np.flip(StructMask, 2)

            # Initialize the mask array on the first iteration
            if i == 0:
                mask = np.zeros(ring_pCT.shape)

            # Assign structure numbers to the mask
            mask[StructMask == 1] = StructureNumbers[i]

        # Save the structure mask
        try:
            os.makedirs(Output_Path)
        except FileExistsError:
            pass
        save(mask, Output_Path+"\labelMask.mha", Header)


#%%
InputFolder="D:/07_Autocontouring/TestScript/ln8"
LN_images=["img_1.3.46.423632.33653620231115125915330.13","img_1.3.46.423632.33653620231115125120905.3","img_1.3.46.423632.33653620231115125523760.8"] #names of the images left border scan, middel scan and right border scan
LN_start=[21,92,171] #for cutting the image where the phantom is on the three scans
LN_stop=[92,174,246]

OutputFolder="D:/07_Autocontouring/TestScript"
OutputName="Processed_Waterphantom.mha"
#

Waterphantom_Path=OutputFolder+"/"+OutputName
pCT_Path="D:/07_Autocontouring/TestScript/Auto_Elekta_CE35_01"
ContourPath="D:/07_Autocontouring/TestScript/Auto_Elekta_CE35_01"
CTV_StructureName="CTV"
StructuresSegmented=["CTV","Bladder","Rectum"]
Factor=4
Output_Path=OutputFolder+"/Ph_sCBCT"
StructureNumbers=[3,1,2]


#%%
Preprocess_WaterPhantom(InputFolder, LN_images, LN_start, LN_stop, OutputFolder,OutputName)
CT2Ph_sCBCT(Waterphantom_Path, pCT_Path, ContourPath, CTV_StructureName, StructuresSegmented, Factor, Output_Path, StructureNumbers)
CT2Ph_sCBCT(Waterphantom_Path, pCT_Path, ContourPath, CTV_StructureName, StructuresSegmented, Factor, Output_Path, StructureNumbers)