import plotly.graph_objects as go
from ipywidgets import widgets,Layout
from matplotlib import cm, colors
from IPython.display import display
from .color import ColorView

class UmapExplorerView():
    controller=None
    fig = None
    def __init__(self,controller):
        self.controller =controller
        self.setup_figure()
        self.setup_ui()
        self.display_gui()
        self.color = ColorView(self)
        pass

    def setup_figure(self):
        # Initialize scatter object.
        customdata = [[None,None,None,None,None]]
        scatter_data = go.Scatter3d(
            x=None, y=None, z=None,
            mode='markers',
            marker=dict(
                size=1.5,
                opacity=0.7,),
            name = 'Umap Frames',
            customdata = customdata,
            hovertemplate="<br>".join([ "Cue Set: %{customdata[4]}", "Trial Type: %{customdata[2]}",
                "Area: %{customdata[0]}", "Position: %{customdata[3]}", "Trial: %{customdata[1]}",
            ])
        )
        colorbar1_data = go.Scatter3d(name = 'colorbar 1',marker_showscale=False,x=[0],y=[0],z=[0],mode='markers',
                        marker=dict(color= [0], size=0,opacity=0.0001,colorscale='Viridis', 
                        colorbar=dict(thickness=20, x=1, title='test',title_side='right',
                        title_font_size=18,lenmode='fraction', len=0.5,yanchor='bottom',y=0,)))
        colorbar2_data = go.Scatter3d(name = 'colorbar 2',marker_showscale=False,x=[0],y=[0],z=[0],mode='markers',
                        marker=dict(color= [0], size=0,opacity=0.0001,colorscale='Blues',
                        colorbar=dict(thickness=20, x=1.1, title='test',title_side='right',
                        title_font_size=18,lenmode='fraction', len=0.5,yanchor='bottom',y=0)))
        self.fig = go.FigureWidget(data=[scatter_data,colorbar1_data,colorbar2_data])
        # format axis appearance
        self.fig.update_layout(showlegend=False,width=1000,height=650,template='plotly_dark',margin=dict(r=10, l=10,b=10, t=10),
                            scene=dict(xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False,
                                    xaxis_title="UMAP1",yaxis_title="UMAP2",zaxis_title="UMAP3"))

    def setup_ui(self):

        ## Output window.
        self.out = widgets.Output()

        # View tab.
        # select color scheme
        color_options = ['Trial Type and Area', 'Trial Type and Position', 'Trial Type and Trial Number','Trial Type and Lick Rate','Trial Type and Speed','Trial Type','Cue Set','Time to Reward','Speed']
        self.color_scheme_selector = widgets.Dropdown( options=color_options, value=color_options[0], description='Color:',layout=Layout(width='300px'),indent=False) 
        # Speed tab.
        self.speed_display_range_control = widgets.FloatRangeSlider( value=[0, 30], min=0, max=50.0, step=0.5, description='Display Range:', continuous_update=False,)
        self.speed_display_range_control.observe(self.controller.update_speed_range, names='value')
        self.speed_window_control =widgets.FloatSlider( value=1.0, min=0.25, max=10.0, step=0.25, description='Smoothing Window:',continuous_update=False,)
        self.speed_window_control.observe(self.controller.update_speed, names='value')
        speed_tab = widgets.VBox([self.speed_display_range_control,self.speed_window_control ])      

        # reward tab.
        self.reward_display_range_control = widgets.FloatRangeSlider( value=[-5, 5], min=-10, max=10.0, step=0.5, description='Reward Range:', continuous_update=False,)
        self.reward_display_range_control.observe(self.controller.update_reward_range, names='value')
        reward_tab = widgets.VBox([self.reward_display_range_control ])    

        # lick tab.
        self.lick_display_range_control = widgets.FloatRangeSlider( value=[0, 5], min=0, max=10.0, step=0.5, description='Lick Range (per sec):', continuous_update=False,)
        self.lick_display_range_control.observe(self.controller.update_lick_range, names='value')
        lick_tab = widgets.VBox([self.lick_display_range_control ])  

        # Animation tab
        self.animation_distance = widgets.FloatText(value=2,description ='Distance')
        self.animation_height = widgets.FloatText(value=0.5,description ='Height')
        self.animation_degree_interval = widgets.IntText(value=2,description ='Degree Interval')
        self.animation_output_scale = widgets.IntText(value=1,description ='Output Scale')
        self.animation_frame_duration = widgets.IntText(value=50,description ='Duration Gif Frame')
        self.animation_file_name = widgets.Text(description='File Name',value='animation')
        self.animation_save_gif_button = widgets.Button(description='Save Gif',button_style='',disabled=False,)
        self.animation_save_view_button = widgets.Button(description='Save View',button_style='',disabled=False,)
        self.animation_play = widgets.Play( value=0, min=0, max=360, step=1,description="Play",disabled=False)
        self.animation_slider = widgets.IntSlider( min=0, max=360)
        widgets.jslink((self.animation_play, 'value'), (self.animation_slider, 'value'))
        animation_tab = widgets.VBox([self.animation_file_name,self.animation_distance, self.animation_height, self.animation_degree_interval, self.animation_output_scale, self.animation_frame_duration,
            self.animation_play,self.animation_slider,self.animation_save_gif_button,self.animation_save_view_button])
        self.animation_play.observe(self.controller.update_animation,names='value')
        self.animation_save_gif_button.on_click(self.controller.save_animation)
        self.animation_save_view_button.on_click(self.controller.save_view)

        # Legend tab
        self.legend_show_areas = widgets.Checkbox(value=True,description='Show Areas')
        legend_tab = widgets.VBox([self.legend_show_areas])
        self.legend_show_areas.observe(self.controller.update_show_areas,names='value')

        # observes
        self.color_scheme_selector.observe(self.controller.update_colormap, names='value')
        # accordion
        accordion = widgets.Accordion(children=[speed_tab,reward_tab,lick_tab,animation_tab,legend_tab],selected_index = None)
        accordion.set_title(0,'Speed')
        accordion.set_title(1,'Reward')
        accordion.set_title(2,'Lick')
        accordion.set_title(3,'Animation')
        accordion.set_title(4,'Legends')
        color_scheme_container = widgets.VBox([self.color_scheme_selector,accordion])
        view_tab = widgets.VBox([color_scheme_container])

        # Setup tabs
        tab = widgets.Tab()
        tab.children = [view_tab]
        tab.set_title(0,'View')
        # organize full ui.
        self.ui = tab

    def display_gui(self):
        display(widgets.HBox([self.ui,self.fig]))
        display(self.out)