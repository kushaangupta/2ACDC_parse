import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

class UmapExplorerModel():
    controller = None
    def __init__(self,umap,labels):
        # Check umap data.
        if umap.ndim!=2: raise NameError("Expected umap to have 2 dimensions")
        if umap.shape[1]!=3: raise NameError('Expected umap dimensions to equal 3')
        # Zero.
        for i in range(3):
            umap[:,i] -=np.min(umap[:,i] )
        # Check labels.
        for key in ['time','frame','position','trial_number','reward_id','set','has_lick','has_reward','session']:
            if key not in labels: raise NameError(f"Expected labels to have field {key}")
        # store.
        self.umap = umap
        self.labels = labels
        # process
        # make trial number a int
        with tqdm(total=100,desc='Processing') as pbar:
            self.labels.trial_number = self.labels.trial_number.astype('int')
            pbar.update(1)
            self.get_position_labels()
            pbar.update(9)
            self.get_lick_rate()
            pbar.update(10)
            self.get_time_to_reward()
            pbar.update(10)
            self.get_speed()
            pbar.update(70)
        pass

    def get_position_labels(self):
        # mark position.
        markers = [{'name':'Initial','position': [0,30]},
               {'name':'Pre-Indicator','position': [30,60]},
               {'name':'Indicator','position': [60,100]},
               {'name':'Pre-R1','position': [100,130]},
               {'name':'R1','position': [130,150]},
               {'name':'Pre-R2','position': [150,180]},
               {'name':'R2','position': [180,200]},
               {'name':'Post-R2','position': [200,230]},]
        for marker in markers:
            self.labels.loc[(self.labels.position.between(marker['position'][0],marker['position'][1])), 'position_label']=marker['name']

    def get_lick_rate(self, window_size = 2):
        # copy relevant data.
        df = self.labels[['time','has_lick']].copy()
        df['center'] = df.rolling(f'{int(window_size*2)}s', on='time',center=True,closed='both')['has_lick'].sum()
        df['center'] = df['center'].fillna(0)
        df['back'] = df.rolling(f'{int(window_size)}s', on='time',center=False,closed='left')['has_lick'].sum()
        df['back'] = df['back'].fillna(0)
        self.labels['lick_rate'] = (df['center']-df['back'])/window_size

    def get_time_to_reward(self):
        for session in  self.labels.session.unique():
            session_labels = self.labels.loc[(self.labels.session==session)]
            for trial in session_labels.trial_number.unique():
                trial_labels = session_labels.loc[session_labels.trial_number==trial]
                reward = trial_labels.loc[(trial_labels.has_reward==True) ,'time'].values
                if reward.size>0:
                    time_to_reward = np.array([ (trial_labels.time-value).dt.total_seconds() for value in reward]).min(axis=0) # in case of more then one reward (manual)
                else: time_to_reward = None
                # store.
                self.labels.loc[(self.labels.session==session) & (self.labels.trial_number==trial),'time_to_reward'] = time_to_reward
    
    def get_speed(self, window_size = 1):
        # Calculate speed by session (overlapping time stamps).
        for session in  self.labels.session.unique():
            df = self.labels.loc[(self.labels.session==session)].copy()
            df['moved'] = df['position'].diff()
            df['time_diff'] = df['time'].diff().dt.total_seconds()
            df['speed'] = df['moved']/df['time_diff']
            
            # Calculate forward looking moving average.
            df['center'] = df.rolling(f'{int(window_size*2)}s', on='time',center=True,closed='both')['speed'].sum()
            df['center'] = df['center'].fillna(0)
            df['center_count'] = df.rolling(f'{int(window_size*2)}s', on='time',center=True,closed='both')['speed'].count()
            df['center_count'] = df['center_count'].fillna(0)
            df['back'] = df.rolling(f'{int(window_size)}s', on='time',center=False,closed='left')['speed'].sum()
            df['back'] = df['back'].fillna(0)
            df['back_count'] = df.rolling(f'{int(window_size)}s', on='time',center=False,closed='left')['speed'].count()
            df['back_count'] = df['back_count'].fillna(0)       
            df['forward'] = (df['center']-df['back'])/(df['center_count']- df['back_count'])
            #store
            self.labels.loc[(self.labels.session==session),'speed'] = df['forward'] 