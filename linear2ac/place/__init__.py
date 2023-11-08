from cgi import test
import vr2p.signal
import vr2p.place_1d
import datetime
import numpy as np 
from linear2ac.place import plot
import scipy.signal
import skimage.measure
import skimage.morphology
import pandas as pd
from numba import jit
from tqdm import tqdm
import random
import warnings
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
import pickle
from tqdm.contrib.concurrent import process_map

class Tyche1dConfig:
    spatial_edges = []
    bin_size = []
    # minimum speed for placefield detection.
    min_speed = 5                    # min speed for frame to be considered. in cm/s
    # df calculation.
    df_sigma_baseline = 20           # sigma used in baseline calculation
    df_window_size = 600             # window size used in baseline calculation in frames
    # speed calculation.
    speed_window_size = 100          # rolling window size for speed calculation in ms.
    speed_ignore_threshold = 7.5     # ignore high distance moved per frame (teleports)
    # bin calculation
    bin_smooth_size = 3              # smooth window size in frames.
    bin_base_quantile = 0.25         # quantile value thats used for the baseline
    bin_signal_threshold = 0.25      # percentage of difference between max and baseline
    # placefield
    pf_min_bin_size = 15              # min. size of placefield in cm
    pf_max_bin_size = 120            # max size of placefield in cm
    pf_threshold_factor = 4          # signal inside placefield needs to be higher by this factor compared to outside.
    pf_max_bin_perc_mean = 0.1       # One bin needs to be atleast this percentage of the mean fluorescence.
    pf_min_df_value = 0.075          # max df value in one bin needs to be atleast this value.
    pf_include_all_fields = True     # calculate outside field values by considering ALL possible placefields of cell.
    # signal detection
    ## baseline detection.
    signal_bin_size = 50             # bin this number of consecutive frames
    signal_base_quantile = 0.25      # determine the baseline std by selecting binned frames below this quantile value
    ## event detection
    signal_onset_factor = 5          # threshold deviation factor * std from baseline that determines onset of event
    signal_end_factor = 1            # threshold deviation factor * std from baseline that determines end of event.
    # correlated activity.
    cor_min_event_correlation = 0.2  # Placefield traversals must be associated with event this many times.
    cor_shuffle_block_size = 100     # size of frame chuncks used in shuffle.
    # bootstrap signficance test.
    bootstrap_do_test = True          # perform bootstrap shuffle.
    bootstrap_num_shuffles = 1000          # number of shuffles to perform in analysis.         
    bootstrap_sig_thresholds = 0.05         # signficance threshold used in shuffle.


""" Tyche 1d Placefield detection protocol.
"""
class Tyche1dProtocol():

    config = Tyche1dConfig()

    def clean_up(self):
        """Clear large memory variables.
        """
        self.F = []
        self.dF = []
        self.F0 = []
        self.event_mask = []

    def detect(self, F, vr, bin_size, selected_frames,verbose=False,track_size = 230, use_parallel = False, parallel_processes=5):
        ## 
        # Store spatial parameters.
        ##
        self.bin_size = bin_size
        self.selected_frames = selected_frames
        self.spatial_edges = np.arange(0,track_size + bin_size, bin_size)
        self.mid_points = self.spatial_edges[1:]-(bin_size/2) # mid points of spatial bins.
        self.config.bin_size = bin_size
        self.config.spatial_edges = self.spatial_edges

        ##
        # Process VR position info.
        ##
        if verbose: print(f'{self._timestamp()}Processing position data..')
        # get position data.
        self.position = vr.path.frame.reset_index()
        # calculate speed.
        self.position['speed'] = self.position.vr2p.rolling_speed(
            window_size = self.config.speed_window_size, ignore_threshold = self.config.speed_ignore_threshold)
        # filter frames with min speed and in 'selected_frames'
        filtered_frames = self.position.loc[(self.position['speed']>=self.config.min_speed) & self.position['frame'].isin(self.selected_frames),'frame']
        self.filtered_pos_data = self.position.loc[self.position['frame'].isin(filtered_frames),['frame','position','trial_number']]

        ##
        # Pre-Process Fluorescence data.
        ##
        if verbose: print(f'{self._timestamp()}Calculating DF/F0..')
        # Get fluorescence data.
        F = F[:]
        # Subtract minimum value to avoid negative dF/F0 values.
        self.F =  F-np.min(F,axis=1)[..., np.newaxis]
        # calculate df (Based on all frames.)
        self.dF, self.F0 = vr2p.signal.df_over_f0(F,'maximin',subtract_min=True,
            sigma_baseline=self.config.df_sigma_baseline, window_size = self.config.df_window_size)
        # Detect calcium events.
        if verbose: print(f'{self._timestamp()}Detecting calcium events..')
        self.event_mask,self.std_quant = vr2p.signal.find_calcium_events(self.dF, bin_size = self.config.signal_bin_size, base_quantile = self.config.signal_base_quantile,
                                             onset_factor = self.config.signal_onset_factor, end_factor = self.config.signal_end_factor)

        ##
        # Bin and find placefields.
        ##
        if verbose: print(f'{self._timestamp()}Bin and find putative placefields..')
        result = bin_and_find_place_fields(self.filtered_pos_data, self.dF, self.event_mask,self.config)
        # store values (written out for clarity).
        self.binF_mat = result['binF_mat']                          # binned fluorescence data (shape: num trial * num bins * num cells)
        self.event_mask_mat = result['event_mask_mat']              # binned event data (shape: num trial * num bins * num cells)
        self.binF = result['binF']                                  # trial averaged fluorescence data (shape: num cells * num bins)
        self.smooth_binF = result['smooth_binF']                    # smoothed trial averaged fluorescence data (shape: num cells * num bins)
        self.thres_binF = result['thres_binF']                      # thresholded binF (as percentage difference from baseline to peak.)(shape: num cells * num bins)
        self.putative_label_im = result['putative_label_im']        # putative place fields labeled image (shape: num cells * num bins)
        self.putative_field_props = result['putative_field_props']  # field properties of putative place fields. (shape: number of placefields)
        self.has_pf = result['has_pf']                              # boolean list if cell had a placefield that passed all the checks.
        # get label image of passed cells.
        self.passed_label_im = self.label_im_from_field_props(result['putative_field_props'],self.has_pf)
        num_cells_with_field = sum(np.amax(self.passed_label_im,1)>0)
        if verbose: print(f'{self._timestamp()}Before shuffle: found {np.nanmax(self.passed_label_im)} place fields across {num_cells_with_field} cells')

        ##
        # Bootstrap.
        #
        # Bootstrap a test distribution by shuffling blocks of frames relative to position info.
        # For each shuffle the same field detection analysis is repeated as before and the occurence of a 'false positive' field is
        # noted for each cell. The test statistics is the percentage of shuffles that have a detected 'false positive place field.
        ##
        if self.config.bootstrap_do_test & num_cells_with_field>0:
            if verbose: print(f'{self._timestamp()}Bootstrap for test statistics..')
            # chunk data.
            chunk_dF = frame_chunks_to_list(self.dF, self.config)
            chunk_event = frame_chunks_to_list(self.event_mask, self.config)
            # bootstrap
            np.random.seed(10)
            indices_list = np.vstack([np.random.choice(len(chunk_dF), len(chunk_dF),replace=False) for i in range(self.config.bootstrap_num_shuffles)]) # key on how to shuffle chunk order (shape num shuffles * num chunks)

            if use_parallel:
                func_run = partial(run_shuffle_test,indices_list=indices_list, chunk_dF=chunk_dF, chunk_event=chunk_event,
                    filtered_pos_data=self.filtered_pos_data, config=self.config)
                shuffle_has_pf = process_map(func_run, range(self.config.bootstrap_num_shuffles), max_workers=parallel_processes)
            else:
                shuffle_has_pf = []
                for i in tqdm(range(self.config.bootstrap_num_shuffles),desc='Bootstrap:'):
                    # store if cell had placefield.
                    shuffle_has_pf.append(run_shuffle_test(i,indices_list, chunk_dF, chunk_event, self.filtered_pos_data, self.config))

            # get p values.
            shuffle_has_pf = np.array(shuffle_has_pf)
            self.p_value = np.sum(shuffle_has_pf,axis=0)/shuffle_has_pf.shape[0]
        else:
            self.p_value = np.zeros(self.binF.shape[0],float) # just set test value to 0 in case no bootstrap test was requested.
        self.is_significant = self.has_pf & (self.p_value<=self.config.bootstrap_sig_thresholds)
        # get signficant label image.
        self.significant_label_im = self.label_im_from_field_props(result['putative_field_props'],self.is_significant)
        if verbose: print(f'{self._timestamp()}After shuffle: found {np.nanmax(self.significant_label_im)} place fields across {sum(self.is_significant)} signficant cells')
        # store placefield info.
        self.pf_putative = vr2p.place_1d.PlaceFields1d(self.putative_label_im, self.smooth_binF,bin_size=bin_size)          # first round: putative placefields.
        self.pf_passed = vr2p.place_1d.PlaceFields1d(self.passed_label_im , self.smooth_binF,bin_size=bin_size)             # second round: placefields that passed all checks.
        self.pf_significant = vr2p.place_1d.PlaceFields1d(self.significant_label_im , self.smooth_binF,bin_size=bin_size)    # third round: signficant place-cells only.
        if verbose: print(f'[{str(datetime.datetime.now().strftime("%H:%M:%S"))}]Done!')
        return

    # helper functions.
    def label_im_from_field_props(self,field_props,selected_cells):
        """Create labeled placefield image from field properties (generated from test_field_props) of selected cells

        Args:
            field_props (dictionary): field properties generated by test_field_props
            selected_cells (list): Boolean list of whether to include palcefields from cell or not

        Returns:
            numpy array: integer placefield label image (shape: num cells * num bins)
        """
        label_im = np.zeros(self.binF.shape,np.int16)
        for prop in field_props:
            # test that cell is a placecell and field passed all checks.
            if selected_cells[prop['cell']] & prop['passed']:
                # fill in mask.
                label_im[prop['coords'][:,0],prop['coords'][:,1]]=np.max(label_im)+1
        return label_im

    # create a timestamp string.
    def _timestamp(self):
        return f'[{str(datetime.datetime.now().strftime("%H:%M:%S"))}]'
    # Plots.

    # Plot dF traces individual neurons.
    def plot_df_traces(self):
        plot.plot_df_traces(self.F,self.dF,self.F0)
    # Plot speed animal vs frame.
    def plot_speed(self):
        plot.plot_speed(self.position, self.config.speed_window_size)
    # plot dF traces of individual neurons with overlayed detected calcium event.
    def plot_detected_events(self):
        plot.plot_detected_events(self.dF, self.event_mask,self.std_quant,
            self.config.signal_onset_factor,self.config.signal_end_factor)
    # Histogram of putative field sizes.
    def plot_putative_field_sizes(self):
        plot.plot_putative_field_sizes(pd.DataFrame(self.putative_field_props), self.config.pf_min_bin_size, self.config.pf_max_bin_size)
    # Histogram of max dF value in putative fields.
    def plot_putative_max_df(self):
        plot.plot_putative_max_df(pd.DataFrame(self.putative_field_props), self.config.pf_min_df_value)
    # Scatter plot outside dF/FO threshold vs Mean inside dF/F0 value
    def plot_putative_inside_vs_outside(self):
        plot.plot_putative_inside_vs_outside(pd.DataFrame(self.putative_field_props), self.config.pf_threshold_factor)
    # Plot binned dF per cell and putative vs passed fields.
    def plot_binned_df_threshold(self):
        plot.plot_binned_df_threshold(self.smooth_binF, self.thres_binF,self.passed_label_im,pd.DataFrame(self.putative_field_props),self.mid_points)
    # plot histogram of putative fields and their calcium event coincendence/correlation.
    def plot_hist_correlation_traversals(self):
        plot.plot_hist_correlation_traversals(pd.DataFrame(self.putative_field_props),self.config)

def run_shuffle_test(shuffle_ind, indices_list, chunk_dF, chunk_event, filtered_pos_data, config):
    """ Run single itteration shuffle of frame data and detect 'spontatneous' placefields in shuffled data.

    Args:
        shuffle_ind (int): Row index of indices_list to use for shuffle (done this way for paralelization).
        indices_list (list): list of arrays that contain different shuffle itterations (num shuffles x num chunks)
        chunk_dF (list): list of numpy arrays representing 'chunked' dF data.
        chunk_event (list): list of numpy arrays representing 'chunked' calcium event data.
        filtered_pos_data (DataFrame): VR spatial information
        config (dictionary): dictionary of parameter settings.

    Returns:
        boolean list: if cell contained a placefield.
    """
    # Shuffle data.
    indices = indices_list[shuffle_ind,:]
    current_dF = np.concatenate([chunk_dF[i] for i in indices],axis=1)
    current_event = np.concatenate([chunk_event[i] for i in indices],axis=1)
    # bin and find placefields.
    shuffle_result = bin_and_find_place_fields(filtered_pos_data, current_dF, current_event, config)
    return shuffle_result['has_pf']

def bin_and_find_place_fields(pos_data, dF,event_mask, config):
    """Bin fluorescence and event data based on spatial information and characterize all placefields.

    Args:
        pos_data (DataFrame): VR spatial information
        dF (numpy array): Fluorescence data (shape: num cells * num frames)
        event_mask (numpy array): Calcium event data (shape: num cells * num frames)
        config (dictionary): dictionary of parameter settings.

    Returns:
        dictionary: results of placefield detection.
        keys:
            ['binF_mat']                # binned fluorescence data (shape: num trial * num bins * num cells)
            ['event_mask_mat']          # binned event data (shape: num trial * num bins * num cells)
            ['binF']                   # trial averaged fluorescence data (shape: num cells * num bins)
            ['smooth_binF']             # smoothed trial averaged fluorescence data (shape: num cells * num bins)
            ['thres_binF']              # thresholded binF (as percentage difference from baseline to peak.)
            ['putative_label_im']       # considered place fields labeled image
            ['putative_field_props']    # field properties of putative place fields.
            ['has_pf']                  # boolean list if cell had a placefield that passed all the checks.
    """
    ##
    # Spatial binning.
    ##
    # Spatial binning of activity data (shape:num_trial * num_bins * num_cells).
    binF_mat, event_mask_mat = get_bin_dF_and_event(pos_data, dF, event_mask, config) # get binned fluorescence and event data 
    # average across trials. 
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        binF = np.transpose(np.nanmean(binF_mat,axis=0)) # num cells * num bins
    # smooth
    smooth_binF = scipy.signal.convolve2d(binF,np.ones((1, config.bin_smooth_size))/config.bin_smooth_size, mode='same',boundary='symm')

    ##
    # Find Place fields.
    ##
    # Threshold bins (as percentage difference from baseline to peak.)
    thres_binF = vr2p.signal.quantile_max_treshold(smooth_binF, config.bin_base_quantile,config.bin_signal_threshold)           
    # Find connected regions
    putative_label_im,_ = scipy.ndimage.label(thres_binF,[[0,0,0],[1,1,1],[0,0,0]])
    # get field properties.
    putative_field_props, has_pf = test_field_props(putative_label_im, smooth_binF, event_mask_mat, config)

    return {'binF_mat':binF_mat, 'event_mask_mat':event_mask_mat,
            'binF': binF, 'smooth_binF':smooth_binF, 'thres_binF':thres_binF,
            'putative_label_im':putative_label_im, 'putative_field_props':putative_field_props, 'has_pf':has_pf}

def get_bin_dF_and_event(pos_data, F, event_mask, config):
    """Takes position dataframe and calculates binned fluorescence data matrix and boolean event mask matrix.

    Args:
        pos_data (DataFrame): VR position data must contain columns 'position', 'trial_number', and 'frame'
        F (numpy array): Fluorescence data, shape: num_cells*num_frames
        event_mask (numpy array): BOolean event data , shape: num_cells*num_frames
        config (dictionary): dictionary of parameter settings.

    Returns:
        numpy array, numpy array: mean fluorescence data (shape:num_trial * num_bins * num_cells),
                                    boolean event data (shape:num_trial * num_bins* num_cells)
    """
    bin_pos = pos_data.copy()
    # change trials to rank order (for filling out matrix).
    bin_pos['trial_number'] = (bin_pos['trial_number'].rank(method='dense')-1).astype(int)
    # Assign bin position.
    bin_pos['bin'] = pd.cut(bin_pos.position,config.spatial_edges, include_lowest=True,labels=False).to_numpy().astype(int)
    bin_pos = bin_pos.loc[bin_pos.bin>=0] # valid bins only.
    # prepare binF matrix.
    num_trials = bin_pos.trial_number.max()+1
    num_bins = config.spatial_edges.size-1
    num_cells = F.shape[0]
    if np.isnan(num_trials): num_trials=0
    binF_mat = np.full((num_trials,num_bins,num_cells),np.nan) # holds mean fluorescence data.
    event_mat = np.full((num_trials,num_bins,num_cells),np.nan,dtype=bool) # holds binary event data.
    # Get frames per trial and bin.
    bin_pos = bin_pos.groupby(['trial_number','bin']).agg({'frame':list})
    # get mean dF value for each bin, trial frame set.
    for index, row in bin_pos.iterrows():
        binF_mat[index[0],index[1],:] = np.mean(F[:,row['frame']],axis=1)
        event_mat[index[0],index[1],:] = np.max(event_mask[:,row['frame']],axis=1)
    return binF_mat, event_mat

def test_field_props(label_im, binF, event_mat, config):
    """Checks properties of labeled placefields.

    Args:
        label_im (numpy array): array with labelled placefields (shape: num_cells * num_spatial_bins)
        binF (numpy array): array with binned fluoresence data (shape: num_cells * num_spatial_bins)
        event_mat (numpy array): array with detected calcium events (shape:num_trial * num_bins * num_cells)
        config (dictionary): dictionary of parameter settings.

    Returns:
        dictionary, placefield properties 'cell': associated cell, 'coords': pixels coordinates of placefields,
                                    'bbox': bounding box of placefield,'size': size in cm of placefield,'max_inside_int': maximum bin value inside the placefield,
                                    'mean_inside_int': mean intensity inside of placefield,'outside_threhsold': outside placefield intensity threshold, 'mean_cell_int':average fluorescence intensity of the cell,
                                    'checks': list of placefield checks,'passed':bool if all checks were passed
        list, bool list of if cell has placefield or not that passed the checks.
    """
    props = np.array(skimage.measure.regionprops(label_im, binF, cache=False))
    # Go through putative place fields.
    field_props = []
    for prop in props:
        coords = prop["coords"]
        area = prop['area']*config.bin_size
        max_inside_int = prop['max_intensity']
        mean_inside_int = prop['mean_intensity']
        cell = coords[0,0]
        bbox = [coords[0,1],coords[-1,1]]
        mean_cell_int = np.nanmean(binF[cell,:])
        # Check how to calculate outside field.
        if config.pf_include_all_fields:
            thres_binF = label_im>0
            inside_mask = thres_binF[cell,:]
        else:
            inside_mask = np.zeros(binF.shape[1],np.bool)
            inside_mask[coords[:,1]] = True
        outside_mask = ~inside_mask
        # Calculate outside field threshold.
        F_out = np.nanmean(binF[cell,outside_mask])
        outside_threshold = (F_out*config.pf_threshold_factor)
        # get number of events associated with traversal.
        if bbox[1]!=bbox[0]: # in case of one pixel placefields.
            events_traversal = np.nanmax(event_mat[:,bbox[0]:bbox[1],cell],axis=1)
        else:
            events_traversal =event_mat[:,bbox[0],cell]
        num_traversals = sum(~np.isnan(events_traversal))
        num_events = sum(events_traversal==True)
        event_correlation = num_events/num_traversals
        # test if placefield passed.
        check_list = {
            'min_field_size':area>config.pf_min_bin_size, # Field needs to be larger than x cm
            'max_field_size':area<=config.pf_max_bin_size, # Field needs to be smaller than x cm
            'inside_vs_outside_int': mean_inside_int>=outside_threshold ,# Mean inside intensity is atleast X times outside intensity.
            'max_bin_value': max_inside_int>=config.pf_min_df_value, # one bin in placefield needs to be atleast this value.
            'max_bin_to_mean': max_inside_int>=(mean_cell_int*config.pf_max_bin_perc_mean ), # one bin in placefield is atleast #% of mean cell fluorescence 
            'event_correlation': event_correlation>=config.cor_min_event_correlation,  # traversals through placefield need to be associated with a signficant event atleast X% of the time.
                    } 
        checks = [check_list[key] for key in check_list.keys()]
        # store stats.
        field_props.append({'cell':cell,'coords':coords,'bbox':bbox, 'size':area, 'max_inside_int':max_inside_int,
                    'mean_inside_int':mean_inside_int, 'outside_threshold':outside_threshold,'mean_cell_int':mean_cell_int,
                    'event_correlation': event_correlation, 'check_list':check_list,'passed':all(checks)})
    # Check which cells have placefield (convenience for bootstrapping).
    unique_pcs = np.unique(np.array([prop['cell']for prop in field_props if prop['passed']]))
    has_pf = np.array([np.isin(cell,unique_pcs) for cell in range(binF.shape[0])])
    return field_props, has_pf

def frame_chunks_to_list(frame_data,config):
    """Takes imaging data and creates a list of chunks of frames (chunk size set according to self.config.cor_shuffle_block_size)

    Args:
        frame_data (numpy array): Imaging data shape: num cells * num_frames

    Returns:
        list of numpy arrays: Chunked imaging data shape: size: number of chunks
    """
    # pack data into list for shuffle.
    num_frames = frame_data.shape[1]
    chunk_data = []
    for start_frame in range(0,num_frames,config.cor_shuffle_block_size):
        # set end of block
        end_frame = start_frame+config.cor_shuffle_block_size
        if end_frame>=num_frames: end_frame=num_frames
        # package frames.
        chunk_data.append(frame_data[:,start_frame:end_frame])
    return chunk_data