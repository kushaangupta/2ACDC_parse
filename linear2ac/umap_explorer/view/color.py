from __future__ import annotations
from turtle import color
import numpy as np
from matplotlib import cm, colors
import plotly.graph_objects as go
import seaborn as sns

class ColorView():
    def __init__(self, view):
        self.view = view
        pass

    def set_heatmap_trial_type_and_area(self,trial_type,position_label):
        with self.view.out:
            colors = np.empty(trial_type.shape[0],object)
            colors[:]= '#FFFFFF'
            # set face color.
            color_lut = [{'position_label': 'Initial','reward_id':1,'color': '#fbb4b9'},
                        {'position_label': 'Indicator','reward_id':1, 'color': '#f768a1'},
                        {'position_label': 'R1','reward_id':1, 'color': '#c51b8a'},
                         {'position_label': 'R2','reward_id':1, 'color': '#7a0177'},
                         #
                        {'position_label': 'Initial','reward_id':2,'color': '#bdc9e1'},
                        {'position_label': 'Indicator','reward_id':2, 'color': '#74a9cf'},
                        {'position_label': 'R1','reward_id':2, 'color': '#2b8cbe'},
                         {'position_label': 'R2','reward_id':2, 'color': '#045a8d'}]
            for lut in color_lut:
                colors[(position_label == lut['position_label']) & (trial_type==lut['reward_id'])] = lut['color']
            self.update_colors(colors)
            self.view.fig.update_traces(marker_showscale=False,selector=dict(name="colorbar 1"))
            self.view.fig.update_traces(marker_showscale=False,selector=dict(name="colorbar 2"))    
            if self.view.legend_show_areas.value: 
                self.view.fig.layout.annotations=[
                    go.layout.Annotation(text='Near', font_color='#c51b8a',font_size=20,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.94,),
                    go.layout.Annotation(text='Initial', font_color='#fbb4b9',font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.9,),
                    go.layout.Annotation(text='Indicator', font_color='#f768a1',font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.875,),
                    go.layout.Annotation(text='R1', font_color='#c51b8a',font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.85,),
                    go.layout.Annotation(text='R2', font_color='#7a0177',font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.825,),
                    go.layout.Annotation(text='Far', font_color='#2b8cbe',font_size=20,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.775,),
                    go.layout.Annotation(text='Initial', font_color='#bdc9e1',font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.735,),
                    go.layout.Annotation(text='Indicator', font_color='#74a9cf',font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.71,),
                    go.layout.Annotation(text='R1', font_color='#2b8cbe',font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.685,),
                    go.layout.Annotation(text='R2', font_color='#045a8d',font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.645,),
                    ]
            else:
                self.view.fig.layout.annotations=[]
    
    def set_heatmap_trial_type_and_position(self,trial_type,position):
        self.heatmap_trial_type_and_value(trial_type, position,[0,230],'RdPu','Blues')
        # set colorbars
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Position-Near (cm)', marker_colorscale = 'RdPu',
                                marker_cmin=0, marker_cmax=230, selector=dict(name="colorbar 1"))
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Position-Far (cm)', marker_colorscale = 'Blues',
                                marker_cmin=0, marker_cmax=230, selector=dict(name="colorbar 2"))
        self.clear_annotations()

    def set_heatmap_trial_type_and_trial_number(self,trial_type,trial_numbers):
        self.heatmap_trial_type_and_value(trial_type, trial_numbers,[trial_numbers.min(),trial_numbers.max()],'RdPu','Blues')
        # set colorbars
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Near - Trial #', marker_colorscale = 'RdPu',
                                marker_cmin=trial_numbers.min(), marker_cmax=trial_numbers.max(), selector=dict(name="colorbar 1"))
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Far - Trial #', marker_colorscale = 'Blues',
                                marker_cmin=trial_numbers.min(), marker_cmax=trial_numbers.max(), selector=dict(name="colorbar 2"))
        self.clear_annotations()

    def set_heatmap_trial_type(self,trial_type):
        color_values = np.full(trial_type.shape[0],'#c51b8a')
        color_values[trial_type == 1]='#2b8cbe'
        self.update_colors(color_values)    
        # Set legend box.
        self.view.fig.layout.annotations=[
            go.layout.Annotation(text='Near', font_color='#c51b8a',font_size=21,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.9,),
            go.layout.Annotation(text='Far', font_color='#2b8cbe',font_size=21,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.87,)]
        # set colorbars
        self.view.fig.update_traces(marker_showscale=False,selector=dict(name="colorbar 1"))
        self.view.fig.update_traces(marker_showscale=False,selector=dict(name="colorbar 2")) 


    def set_heatmap_trial_type_and_lick_rate(self,trial_type,lick_rate):
        display_range = self.view.lick_display_range_control.value
        self.heatmap_trial_type_and_value(trial_type, lick_rate,display_range,'RdPu','Blues')
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Near - Lick Rate (per s)', marker_colorscale = 'RdPu',
                                marker_cmin=display_range[0], marker_cmax=display_range[1], selector=dict(name="colorbar 1"))
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Far - Lick Rate (per s)', marker_colorscale = 'Blues',
                                marker_cmin=display_range[0], marker_cmax=display_range[1], selector=dict(name="colorbar 2")) 
        self.clear_annotations()

    def set_heatmap_trial_type_and_speed(self,trial_type,speed):
        display_range = self.view.speed_display_range_control.value
        self.heatmap_trial_type_and_value(trial_type, speed,[display_range[0],display_range[1]],'RdPu','Blues')
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Near - Speed (cm/s)', marker_colorscale = 'RdPu',
                                marker_cmin=display_range[0], marker_cmax=display_range[1], selector=dict(name="colorbar 1"))
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Far - Speed (cm/s)', marker_colorscale = 'Blues',
                                marker_cmin=display_range[0], marker_cmax=display_range[1], selector=dict(name="colorbar 2")) 
        self.clear_annotations()

    def set_heatmap_cue_set(self,cue_set):
        color_values = np.full(cue_set.shape[0],'#FFFFFF')
        color_list = {
        'Cue Set A': colors.rgb2hex(sns.color_palette("colorblind")[0]),
        'Cue Set B': colors.rgb2hex(sns.color_palette("colorblind")[1]),
        'Cue Set B2': colors.rgb2hex(sns.color_palette("colorblind")[1]),
        'Cue Set C': colors.rgb2hex(sns.color_palette("colorblind")[2]),
        'Cue Set D': colors.rgb2hex(sns.color_palette("colorblind")[3]),
        'Cue Set E': colors.rgb2hex(sns.color_palette("colorblind")[4]),
        }
        for set in color_list.keys():
            color_values[cue_set==set] = color_list[set]
        self.update_colors(color_values)  
        # annotations
        self.view.fig.layout.annotations = [
            go.layout.Annotation(text=set, font_color=color_list[set],font_size=16,align='left', showarrow=False, xref='paper', yref='paper', x=0.1, y=0.9-(0.025*i),)
            for i, set in enumerate(color_list.keys())]
        # set colorbars
        self.view.fig.update_traces(marker_showscale=False,selector=dict(name="colorbar 1"))
        self.view.fig.update_traces(marker_showscale=False,selector=dict(name="colorbar 2")) 

    def set_heatmap_time_to_reward(self, time_to_reward):
        display_range = self.view.reward_display_range_control.value
        self.update_colors_on_property(time_to_reward, display_range, 'RdBu')
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Time to Reward', marker_colorscale = 'RdBu',
                                marker_cmin=display_range[0], marker_cmax=display_range[1], selector=dict(name="colorbar 1"))
        self.view.fig.update_traces(marker_showscale=False,selector=dict(name="colorbar 2")) 
        self.clear_annotations()

    def set_heatmap_speed(self, speed):
        display_range = self.view.speed_display_range_control.value
        self.update_colors_on_property(speed, display_range, 'viridis')
        self.view.fig.update_traces(marker_showscale=True,marker_colorbar_title_text = 'Speed (cm/s)', marker_colorscale = 'viridis',
                                marker_cmin=display_range[0], marker_cmax=display_range[1], selector=dict(name="colorbar 1"))
        self.view.fig.update_traces(marker_showscale=False,selector=dict(name="colorbar 2")) 
        self.clear_annotations()

    def update_colors_on_property(self,values,display_range,cmap_name):
        norm = colors.Normalize(vmin=display_range[0], vmax=display_range[1])
        cmap = cm.get_cmap(cmap_name)
        color_values = list(map(colors.rgb2hex,cmap(norm(values)))) 
        self.update_colors(color_values)

    def heatmap_trial_type_and_value(self,trial_type,values,range_map,colormap_T1,colormap_T2):
        with self.view.out:
            color_values = np.full(values.shape[0],'#FFFFFF')
            norm = colors.Normalize(vmin=range_map[0], vmax=range_map[1])
            for reward_id in [1,2]:
                if reward_id==1: cmap = cm.get_cmap(colormap_T1)
                else: cmap = cm.get_cmap(colormap_T2)
                ind = trial_type==reward_id
                color_values[ind] = list(map(colors.rgb2hex,cmap(norm(values[ind])))) 
            self.update_colors(color_values)
    
    def update_colors(self,colors):
        self.view.fig.update_traces(marker_color = colors, selector=dict(name="Umap Frames"))
    
    def clear_annotations(self):
        self.view.fig.layout.annotations = []