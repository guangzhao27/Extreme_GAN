# -*- coding: UTF-8 -*-
import os
import json
import numpy as np
import torch
import sys
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir, os.pardir))
sys.path.append(parent_directory)

from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.GAN import Discriminator_wassertein, Generator_linear, plot_GAN_loss, fit_gan_wasserstein, build_gan_scenarios
from GEFcom2014 import wind_data, load_data, pv_data
from GEFcom2014.forecast_quality import quantiles_and_evaluation
from GEFcom2014.utils import dump_file
from torch.utils.benchmark import timer

Test_Type = True

# ------------------------------------------------------------------------------------------------------------------
# GEFcom IJF_paper case study
# Solar track: 3 zones
# Wind track: 10 zones
# Load track: 1 zones
# 50 days picked randomly per zone for the VS and TEST sets
#
# A multi-output wasserstein GAN with gradient penalty:
# Generator = a linear generator
# Discriminator = a wasserstein discriminator
# ------------------------------------------------------------------------------------------------------------------

tag = 'wind' # pv, wind, load
gpu = True # put False to use CPU
print('Using gpu: %s ' % torch.cuda.is_available())
if gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_path = 'export/WGANGP_' + tag + '/'

if not os.path.isdir(dir_path):  # test if directory exist
    os.makedirs(dir_path)

# ------------------------------------------------------------------------------------------------------------------
# Built the LS, VS, and TEST sets
# ------------------------------------------------------------------------------------------------------------------

if tag == 'pv':
    # WARNING: the time periods where PV is always 0 (night hours) are removed -> there are 8 periods removed
    # The index of the time periods removed are provided into indices
    data, indices = pv_data(path_name='../../data/solar_new.csv', test_size=50, random_state=0)
    ylim_loss = [-10, 10]
    ymax_plf = 3
    ylim_crps = [0, 10]
    nb_zones = 3


elif tag == 'wind':
    data = wind_data(path_name='../../data/wind_data_all_zone.csv', test_size=50, random_state=0, test_type=Test_Type)
    ylim_loss = [-20, 10]
    ymax_plf = 8
    ylim_crps = [6, 12]
    nb_zones = 10
    indices = []

elif tag == 'load':
    data = load_data(path_name='../../data/load_data_track1.csv', test_size=50, random_state=0)
    ylim_loss = [-40, 10]
    ymax_plf = 2
    ylim_crps = [0, 5]
    nb_zones = 1
    indices = []

# reduce the LS size from 634 days to D days
df_x_LS = data[0].copy()
df_y_LS = data[1].copy()
df_x_VS = data[2].copy()
df_y_VS = data[3].copy()
df_x_TEST = data[4].copy()
df_y_TEST = data[5].copy()

nb_days_LS = len(df_y_LS)
nb_days_VS = len(df_y_VS)
nb_days_TEST = len(df_y_TEST)
print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

# ------------------------------------------------------------------------------------------------------------------
# Scale the LS, VS, and TEST sets
# ------------------------------------------------------------------------------------------------------------------

# WARNING: use the scaler fitted on the TRAIN LS SET !!!!
x_LS_scaled, y_LS_scaled, x_VS_scaled, y_VS_scaled, x_TEST_scaled, y_TEST_scaled, y_LS_scaler = scale_data_multi(x_LS=df_x_LS.values, y_LS=df_y_LS.values, x_VS=df_x_VS.values, y_VS=df_y_VS.values, x_TEST=df_x_TEST.values, y_TEST=df_y_TEST.values)

non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))