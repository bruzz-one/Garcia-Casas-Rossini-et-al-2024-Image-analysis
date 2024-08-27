import numpy as np
import pathlib
from skimage import filters, io
import cv2 as cv
import matplotlib.pyplot as plt
import os
import pandas as pd

data_path=pathlib.Path(r'path\to\images\folder')  

reds=[] 
greens=[]
df=pd.DataFrame(columns=['cell', 'pixels_combined', 'pixels_red' 'value', 'note'])
data_names=np.sort([file for file in os.listdir(data_path) if file.endswith('.tif') and os.path.isfile(os.path.join(data_path, file))])
for i in range(len(data_names)):
    if data_names[i][-8]=='0':
        greens.append(data_names[i])
    else:
        reds.append(data_names[i])

for i in range(len(reds)):
    nota=' '
    item1=list(reds[i])
    item1_removed=item1.pop(-8)
    for k in range(len(greens)):
        item2=list(greens[k])
        item2_removed=item2.pop(-8)
        if (item1==item2)&(item1_removed!=item2_removed):
            print(reds[i], greens[k]+ '\n')
            green_image=io.imread(data_path.joinpath(greens[k]))
            red_image=io.imread(data_path.joinpath(reds[i]))

            red_image_background=red_image
            green_image_background=green_image

            red_image_norm=(red_image_background-np.min(red_image_background))/(np.max(red_image_background)-np.min(red_image_background))
            green_image_norm=(green_image_background-np.min(green_image_background))/(np.max(green_image_background)-np.min(green_image_background))

            red_image_convolved = filters.gaussian(red_image_norm, sigma=1)
            green_image_convolved = filters.gaussian(green_image_norm, sigma=1)

            threshold_red=(np.mean(red_image_convolved[red_image_convolved>0.2]))/2
            threshold_green=(np.mean(green_image_convolved[green_image_convolved>0.2]))/2

            binarized_red=red_image_convolved>threshold_red
            binarized_green=green_image_convolved>threshold_green

            try:
                combined=binarized_red*binarized_green

                fig, axs=plt.subplots(1,3, figsize=(10,10))
                axs[0].imshow(binarized_red, cmap='binary')
                axs[0].set_title('Image binarized red')
                axs[1].imshow(binarized_green, cmap='binary')
                axs[1].set_title('Image binarized green')
                axs[2].imshow(combined, cmap='binary')
                axs[2].set_title('Image binarized combined')
                plt.suptitle(reds[i])
                plt.savefig(f'path\to\image\to\save.png', dpi=300)

                pixels_combined=np.sum(combined)
                pixels_red=np.sum(binarized_red)
                fraction=pixels_combined/pixels_red
                
            except ValueError:
                nota='The two images are of different sizes'
                pixels_combined=0
                pixels_red=0
                fraction=0

            line=pd.DataFrame([[reds[i], pixels_combined, pixels_red, fraction, nota]], columns=['cell', 'pixels_combined', 'pixels_red', 'value', 'note'] )
            df=pd.concat([df,line], ignore_index=True)
df.to_excel(r'path\to\excel\file.xlsx') 
