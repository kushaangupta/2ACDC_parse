import warnings
import numpy as np
import io 
from PIL import Image
import math
class UmapExplorerController():
    model = None
    view = None
    def __init__(self):
        pass

    def update_display_data(self):

        with warnings.catch_warnings(): # cant throw Nan value warning that I want to surpress
            warnings.simplefilter('ignore')
            custom_data = [[ row[key] for key in ['position_label','trial_number','reward_id','position','set']] for index, row in self.model.labels.iterrows()]
            self.view.fig.update_traces(x = self.model.umap[:,0] - np.min(self.model.umap[:,0]),
             y = self.model.umap[:,1],
             z = self.model.umap[:,2], customdata=custom_data,selector=dict(name="Umap Frames"))


    def update_colormap(self,*args):
        with self.view.out:
            scheme_value = self.view.color_scheme_selector.value
            if scheme_value =='Trial Type and Area': 
                self.view.color.set_heatmap_trial_type_and_area(self.model.labels['reward_id'],self.model.labels['position_label'])
            if scheme_value=='Trial Type and Trial Number': 
                self.view.color.set_heatmap_trial_type_and_trial_number(self.model.labels['reward_id'],self.model.labels['trial_number'])
            if scheme_value=='Trial Type and Position':
                self.view.color.set_heatmap_trial_type_and_position(self.model.labels['reward_id'],self.model.labels['position'])
            if scheme_value=='Trial Type':
                self.view.color.set_heatmap_trial_type(self.model.labels['reward_id'])
            if scheme_value=='Trial Type and Lick Rate':
                self.view.color.set_heatmap_trial_type_and_lick_rate(self.model.labels['reward_id'],self.model.labels['lick_rate'])
            if scheme_value=='Trial Type and Speed':
                self.view.color.set_heatmap_trial_type_and_speed(self.model.labels['reward_id'],self.model.labels['speed'])
            if scheme_value=='Cue Set':
                self.view.color.set_heatmap_cue_set(self.model.labels['set'])
            if scheme_value=='Time to Reward':
                self.view.color.set_heatmap_time_to_reward(self.model.labels['time_to_reward'])
            if scheme_value=='Speed':
                self.view.color.set_heatmap_speed(self.model.labels['speed'])
    
    def update_speed_range(self,*args):
        with self.view.out:
            scheme_value = self.view.color_scheme_selector.value
            if scheme_value=='Speed':
                self.view.color.set_heatmap_speed(self.model.labels['speed'])
            if scheme_value=='Trial Type and Speed':
                self.view.color.set_heatmap_trial_type_and_speed(self.model.labels['reward_id'],self.model.labels['speed'])
    
    def update_reward_range(self, *args):
        with self.view.out:
            scheme_value = self.view.color_scheme_selector.value
            if scheme_value=='Time to Reward':
                self.view.color.set_heatmap_time_to_reward(self.model.labels['time_to_reward'])

    def update_lick_range(self, *args):
        with self.view.out:
            scheme_value = self.view.color_scheme_selector.value
            if scheme_value=='Trial Type and Lick Rate':
                self.view.color.set_heatmap_trial_type_and_lick_rate(self.model.labels['reward_id'],self.model.labels['lick_rate'])
    
    def update_speed(self,*args):
        with self.view.out:
            self.model.get_speed(window_size = self.view.speed_window_control.value)
            scheme_value = self.view.color_scheme_selector.value
            if scheme_value=='Speed':
                self.view.color.set_heatmap_speed(self.model.labels['speed'])
            if scheme_value=='Trial Type and Speed':
                self.view.color.set_heatmap_trial_type_and_speed(self.model.labels['reward_id'],self.model.labels['speed'])       
    
    def update_show_areas(self,*args):
        with self.view.out:
            scheme_value = self.view.color_scheme_selector.value
            if scheme_value=='Trial Type and Area':
                self.view.color.set_heatmap_trial_type_and_area(self.model.labels['reward_id'],self.model.labels['position_label'])
    
    def update_animation(self,value,*args):
        with self.view.out:
            self.update_animation_frame(value['new'])
    
    def update_animation_frame(self,degree):
            # get points on circle/
            def PointsInCircum(r,n=360):
                pi = math.pi
                return [(math.cos(2*pi/n*x)*r ,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]
            all_points = PointsInCircum(self.view.animation_distance.value,360)
            #update camera.
            cam_center = self.view.fig.layout.scene.camera.center
            camera = dict(
            eye=dict(x=all_points[degree][0]+cam_center['x'], y=all_points[degree][1]+cam_center['y'], z=self.view.animation_height.value,),
            center = cam_center
                )
            self.view.fig.update_layout(scene_camera=camera)
    def fig2img(self):
        #convert Plotly fig to  an array
        fig_bytes = self.view.fig.to_image(format="png",scale = self.view.animation_output_scale.value)
        buf = io.BytesIO(fig_bytes)
        return Image.open(buf)

    def save_animation(self,*args):
        with self.view.out:

            imgs=[]
            for degree in range(0,360,self.view.animation_degree_interval.value):
                self.update_animation_frame(degree)
                imgs.append(self.fig2img())
            imgs[0].save(f"{self.view.animation_file_name.value}.gif", save_all=True, append_images=imgs[1:],
             duration=50, loop=0)
    
    def save_view(self,*args):
        with self.view.out:
            self.view.fig.write_image(f"{self.view.animation_file_name.value}-view.png",format="png",scale = self.view.animation_output_scale.value)


            
