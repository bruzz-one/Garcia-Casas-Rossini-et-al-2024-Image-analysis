import matplotlib.pyplot as plt
from skimage import measure
import tifffile as tiff
import pandas as pd
import numpy as np
import pathlib
import os

class IntLvls():
    def __init__(self, data_path):
        self.data_path = data_path
        pathlib.Path(data_path.joinpath('results')).mkdir(parents=True, exist_ok=True)
        self.data = None
        self.data_zero = None

    def get_data(self):
        data_names = np.sort([file for file in os.listdir(self.data_path) if file.endswith('.tif') and os.path.isfile(self.data_path.joinpath(file))])
        return data_names
    
    def get_contours(self, data_name):
        self.data = tiff.imread(self.data_path.joinpath(data_name))
        self.data_zero = np.where(np.isnan(self.data), 0, self.data)
        contours = measure.find_contours(self.data_zero[0], 0.1)
        return contours
    
    def get_results(self, contours):
        results = np.empty((len(contours), len(self.data) + 1))
        for k in range(len(contours)):
            y, x = np.meshgrid(contours[k][:, 1], contours[k][:, 0], sparse = False)
            y, x = np.ravel(y.astype(int)), np.ravel(x.astype(int))
            coordinates = np.unique(np.stack((np.ravel(y.astype(int)), np.ravel(x.astype(int)))).T, axis = 0)
            res = np.empty((len(coordinates), 1))
            for i in range(len(coordinates)):
                    res[i] = (self.data[0][coordinates[i, 1], coordinates[i, 0]])
            results[k,0] = len(res) - np.sum(np.isnan(res))
            results[k,1] = np.nanmean(res)
            coordinates = coordinates[np.where(np.isnan(res) == False)[0]]
            for j in range(1, len(self.data)):
                res = np.empty((len(coordinates), 1))
                for i in range(len(coordinates)):
                    res[i] = (self.data[j][coordinates[i, 1], coordinates[i, 0]])
                results[k,j+1] = np.nanmean(res)
        return results
    
    def get_results_clean(self, results, contours):
        results = np.where(np.isnan(results), 0, results)
        results_clean = results[np.where(np.sum(results == 0, axis = 1) == 0)[0]]
        results_clean = np.insert(results_clean, 0, np.arange(0, len(results_clean), 1), axis = 1)
        contours_clean = [contours[i] for i in np.where(np.sum(results == 0, axis = 1) == 0)[0]]
        return results_clean, contours_clean
    
    def save_results(self, results_clean, data_name):
        results_clean = pd.DataFrame(results_clean, columns=['id_contatto', 'Pixel size', *['T_' + str(i+1) for i in range(len(self.data))]])
        pd.DataFrame(results_clean).to_csv(self.data_path.joinpath('results/' + data_name[:-4] + '_RESULTS.csv'), 
                                        sep = ";", 
                                        index = False, 
                                        header = True, 
                                        decimal = ',', 
                                        float_format = '%.3f')
        
    def plot_results(self, results_clean, data_name):
        fig, ax = plt.subplots()
        ax.imshow(results_clean[:, 2:], cmap='viridis', vmin = results_clean[:, 2:].min(axis = 1).mean(), vmax = results_clean[:, 2:].max(axis = 1).mean(), aspect = 'auto')
        ax.set_xlabel('Time')
        ax.set_ylabel('Identified contours')
        plt.savefig(self.data_path.joinpath('results/' + data_name[:-4] + '_INTENSITIES.png'), dpi = 300, bbox_inches = 'tight')

    def plot_contours(self, contours_clean, data_name):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.data_zero[0], cmap=plt.cm.gray)
        for count, contour in enumerate(contours_clean):
            plot = ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            ax.text(contour[:,1].mean()+3, contour[:,0].mean(), str(count), fontsize = 10, color = plot[0].get_color());
        plt.savefig(self.data_path.joinpath('results/' + data_name[:-4] + '_CONTOURS.png'), dpi = 300, bbox_inches = 'tight')
    
    def run(self):
        data_names = self.get_data()
        for data_name in data_names:
            print(data_name)
            contours = self.get_contours(data_name)
            results = self.get_results(contours)
            results_clean, contours_clean = self.get_results_clean(results, contours)
            self.save_results(results_clean, data_name)
            self.plot_results(results_clean, data_name)
            self.plot_contours(contours_clean, data_name)
            
if __name__ == "__main__":
    data_path = pathlib.Path(input('Enter the path to the data:'))
    int_lvl = IntLvls(data_path)
    int_lvl.run()
