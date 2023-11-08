from dataclasses import field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interactive, HBox, VBox,widgets
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

def plot_df_traces(F,dF,F0):
    # Generate graph using Figure Constructor
    layout = go.Layout(
        title="Cell Fluorescence", xaxis_title="Frame", yaxis_title="Intensity",
        xaxis_rangeslider_visible = True, template='simple_white',)

    fig = go.Figure(layout=layout)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]],figure=fig)
    print(dF.shape)
    fig.add_trace(go.Scatter(y=dF[0,:],name='dF',marker_color="green"),secondary_y=True)
    fig.add_trace(go.Scatter(y=F[0,:],name='F',marker_color="black"))
    fig.add_trace(go.Scatter(y=F0[0,:],name='F0',marker_color="blue"))
    # adjust secondary axis appearance.
    fig.update_yaxes(secondary_y=True, title_text="dF/F0",color='green')
    
    f = go.FigureWidget(fig)
    def update_range(ineuron=0):
        f.data[0]['y']= dF[ineuron,:]
        f.data[1]['y']= F[ineuron,:]
        f.data[2]['y']= F0[ineuron,:]
    
    # display the FigureWidget and slider with center justification
    vb = VBox((interactive(update_range,
                            ineuron=(0,F.shape[0])),f ))
    display(vb)

def plot_speed(position, window_size):
    layout = go.Layout(
        title=f'Animal Speed (window: {window_size} ms)',
        xaxis_title="Frame", yaxis_title="Speed (cm/s)",
        xaxis_rangeslider_visible = True, template='simple_white',)
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x = position['frame'], y=position['speed'],name='Animal Speed'))
    fig.show()

def plot_frames_per_bin(pos_data,edges):
    fig, ax = plt.subplots(facecolor=(1, 1, 1))
    ax = pos_data.plot.hist(ax=ax,bins=edges,edgecolor='black',title='Calcium frames in each bin')
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Number of Frames")

def plot_detected_events(dF, event_mask,std_quant,onset_factor,end_factor):
    # Generate graph using Figure Constructor
    layout = go.Layout(
        title="Cell Fluorescence",
        xaxis_title="Frame", yaxis_title="DF/F0",
        xaxis_rangeslider_visible = True, template='simple_white',)
    fig = go.Figure(layout=layout)
    fig = make_subplots(specs=[[{"secondary_y": True}]],figure=fig)
    fig.add_trace(go.Scatter(y=dF[0,:],name='dfF/F0'))
    fig.add_trace(go.Scatter(y=event_mask[0,:],name='Event detection'),secondary_y=True)
    # threshold lines.
    fig.add_hline(y=std_quant[0]*onset_factor, line_width=3,line_color='green')
    fig.add_hline(y=std_quant[0]*end_factor, line_width=3,line_color='red')
    fig.update_yaxes(title_text="Event Detection", secondary_y=True,color='orange',type='category',categoryorder='category ascending')
    f2 = go.FigureWidget(fig)
    def update_range(ineuron=0):
        f2.data[0]['y']=dF[ineuron,:]
        f2.data[1]['y']=event_mask[ineuron,:]
        # threshold lines
        f2.layout.shapes[0]['y0']= std_quant[ineuron]*onset_factor
        f2.layout.shapes[0]['y1']=f2.layout.shapes[0]['y0']
        f2.layout.shapes[1]['y0']= std_quant[ineuron]*end_factor
        f2.layout.shapes[1]['y1']=f2.layout.shapes[1]['y0']
        #f2.layout.yaxis2.range=[False,True]
    # display the FigureWidget and slider with center justification
    vb = VBox((interactive(update_range, ineuron=(0,dF.shape[0])),f2 ))
    display(vb)

def plot_putative_field_sizes(field_prop, min_bin_size,max_bin_size):
    fig, ax = plt.subplots(facecolor=(1, 1, 1))
    n_pass = sum(field_prop['size']>=min_bin_size)
    fig.suptitle(f'Putative place field sizes ({n_pass}/{field_prop.shape[0]})', fontsize=12)
    field_prop['size'].plot.hist(bins=20,edgecolor='black')
    ax.set_xlabel('Field bin size')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axvline(min_bin_size,color='red')
    ax.axvline(max_bin_size,color='red')

def plot_putative_max_df(field_prop,min_df_value):
    fig, ax = plt.subplots(facecolor=(1, 1, 1))
    n_pass = sum(field_prop['max_inside_int']>=min_df_value)
    fig.suptitle(f'Max DF/F0 value in field ({n_pass}/{len(field_prop)})', fontsize=12)
    field_prop['max_inside_int'].plot.hist(bins=200,edgecolor='black')
    ax.set_xlabel('Max DF/F0')
    ax.set_xlim(0,1.5)
    ax.axvline(min_df_value,color='red')

def plot_putative_inside_vs_outside(field_prop,threshold_factor):
    fig, ax = plt.subplots(facecolor=(1, 1, 1))
    n_pass = sum(field_prop['mean_inside_int']>=field_prop['outside_threshold'])
    fig.suptitle(f'Outside threshold factor vs Mean DF/F0 value\nThreshold factor: {threshold_factor} ({n_pass}/{field_prop.shape[0]}) ', fontsize=12)
    ax.scatter(x=field_prop['outside_threshold'],y=field_prop['mean_inside_int'],edgecolor='black',alpha=0.5)
    ax.plot(ax.get_xlim(), ax.get_xlim(),color='red')
    ax.set_ylabel('Mean DF/F0 value')
    ax.set_xlabel('Outside threshold value')

def plot_binned_df_threshold(smooth_binF, thres_binF,passed_label_im,field_prop,mid_points):
    print(mid_points.shape)
    print(smooth_binF.shape)
    def f(icell=0):
        fig, (ax1,ax2) = plt.subplots(2,1,facecolor=(1, 1, 1), constrained_layout=True) # for image copying background is white
        fig.suptitle(f'Cell #{icell}', fontsize=12)
        # before filter.
        ax1.plot(mid_points, smooth_binF[icell,:])
        ax1t = ax1.twinx() 
        ax1t.plot(mid_points, thres_binF[icell,:],axes=ax1t,color='green')
        ax1t.set_ylim(0,1.1)
        ax1.set_xlabel("Position Bin")
        ax1.set_ylabel("Fluorescence")
        ax1t.set_ylabel("Putative place field")
        ax1t.yaxis.label.set_color('green')
        ax1t.tick_params(axis='y', colors='green')
        ax1t.spines["right"].set_edgecolor('green')
        # after filter
        ax2.plot(mid_points, smooth_binF[icell,:])
        ax2t = ax2.twinx() 
        ax2t.plot(mid_points, passed_label_im[icell,:]>0,axes=ax2t,color='green')
        ax2.set_xlabel("Position Bin")
        ax2.set_ylabel("Fluorescence")
        ax2t.set_ylabel("Putative place field")
        ax2t.set_ylim(0,1.1)
        ax2t.yaxis.label.set_color('green')
        ax2t.tick_params(axis='y', colors='green')
        ax2t.spines["right"].set_edgecolor('green')
        ax1.set_title('Before filter')
        ax2.set_title('After filter')
        display(field_prop.loc[field_prop['cell']==icell])
        
    interactive_plot = interactive(f, icell=(0,smooth_binF.shape[0]))
    output = interactive_plot.children[-1]
    display(interactive_plot)

def plot_hist_correlation_traversals(field_props, config):
    fig, ax = plt.subplots(facecolor=(1, 1, 1))
    bin_edges = np.arange(0,1,0.025)
    ax = field_props.plot.hist(y='event_correlation',ax=ax,bins=bin_edges,edgecolor='black',label='Putative')
    ax = field_props.loc[field_props['event_correlation']>config.cor_min_event_correlation,'event_correlation'].plot.hist(ax=ax,bins=bin_edges,edgecolor='black',label='Significant',
        title='Correlation of placefield traversal and calcium events',alpha=0.5)
    ax.set_xlabel("Coincidence (%)")
    ax.legend(["Putative","Signficant"])
