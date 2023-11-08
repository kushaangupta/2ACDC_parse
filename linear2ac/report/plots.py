import pandas as pd
from matplotlib import cm
import numpy as np
def scatter_summary_performance(ax,data):
    # collect data.
    plot_data = []
    for isession, vr in enumerate(data):
        sets = vr.trial.loc[(vr.trial.status!='INCOMPLETE') & vr.trial['is_guided']==False ,'set'].unique()
        for icue, cue in enumerate(sets):
            trials = vr.trial.loc[(vr.trial['is_guided']==False)&(vr.trial['status'].isin(['CORRECT','INCORRECT']))&
                    (vr.trial.set==cue)]
            if not trials.empty:
                accuracy = trials.loc[trials.status=='CORRECT','status'].count()/trials.shape[0]*100
                plot_data.append({'session':isession+1,'set':cue,'accuracy':accuracy,'seti':icue})
    plot_data=pd.DataFrame(plot_data)
    # plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,len(data)+1)
    ax.set_ylim(0,100)
    ax.set_xlabel('Session (#)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Summary Mean Performance')
    # color map.
    sets = plot_data.set.unique()
    col = cm.get_cmap('Accent', len(sets)+1)
    handles = [ax.scatter(plot_data.loc[plot_data.set==cue].session,
                        plot_data.loc[plot_data.set==cue].accuracy,
                        color = col(i),edgecolor='black', clip_on=False,s=70) for i, cue in enumerate(sets)]
    ax.legend(handles,sets,bbox_to_anchor=(1, 1), loc='upper left', ncol=1, prop={'size': 6})

def scatter_summary_max_performance(ax,data, win_size=25,threshold=75):
    # collect data.
    plot_data = []
    for isession, vr in enumerate(data):
        # get all cue sets in session.
        sets = vr.trial.loc[(vr.trial.status!='INCOMPLETE') & vr.trial['is_guided']==False ,'set'].unique()
        for icue, cue in enumerate(sets):
            # select right trials.
            trials = vr.trial.loc[(vr.trial['is_guided']==False)&(vr.trial['status'].isin(['CORRECT','INCORRECT','NO_RESPONSE']))&
                    (vr.trial.set==cue)].copy()
            if not trials.empty:
                trials['correct'] = trials['status']=='CORRECT'
                accuracy = (trials.rolling(win_size,
                    on='trial_number',min_periods=win_size)['correct'].sum()/win_size)*100
                # check that not all values are nan (throws warning otherwise)
                if not np.isnan(accuracy).all():
                    # store data.
                    plot_data.append({'session':isession+1,'set':cue,'accuracy':np.nanmax(accuracy),'seti':icue})
    plot_data=pd.DataFrame(plot_data)
    # plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,len(data)+1)
    ax.set_ylim(0,100)
    ax.set_xlabel('Session (#)')
    ax.set_ylabel(f'Max Accuracy (%)\nRolling Average of {win_size}')
    ax.set_title(f'Summary Max. Performance')
    # color map.
    if not plot_data.empty:
        sets = plot_data.set.unique()
        col = cm.get_cmap('Accent', len(sets)+1)
        handles = [ax.scatter(plot_data.loc[plot_data.set==cue].session,
                            plot_data.loc[plot_data.set==cue].accuracy,
                            color = col(i),edgecolor='black', clip_on=False,s=70) for i, cue in enumerate(sets)]
        ax.legend(handles,sets,bbox_to_anchor=(1, 1), loc='upper left', ncol=1, prop={'size': 6})
        ax.axhline(threshold,color='green')

def plot_summary_total_trials(ax,data):
    # collect data.
    plot_data = []
    for isession, vr in enumerate(data):
        trials = vr.trial.loc[(vr.trial.status!='INCOMPLETE')]
        plot_data.append({'session':isession+1,'trial_number':trials.shape[0]})
    # plot
    plot_data = pd.DataFrame(plot_data)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(1,len(data)+1)
    ax.set_ylim(0,400)
    #plot data.
    plot_data.plot(x='session',y='trial_number',color='black',
                xlabel='Session (#)',ylabel='Total Trials',title='Number of Trials',
                marker='o',ax=ax,legend=False)

def plot_summary_total_water(ax,data):
    # collect data.
    plot_data = []
    for isession, vr in enumerate(data):
        amount = vr.reward.amount.sum()/1000
        per_reward = vr.reward.amount.mean()
        plot_data.append({'session':isession+1,'amount':amount,'reward':per_reward})
    plot_data = pd.DataFrame(plot_data)
    # total water axis.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(1,len(data)+1)
    ax.set_ylim(0,1.25)
    # water per reward.
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.label.set_color('blue')
    ax2.spines["right"].set_edgecolor('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax2.set_ylim(0,12.5)
    #plot total water data.
    plot_data.plot(x='session',y='amount',color='black',
                xlabel='Session (#)',ylabel='Total Water (ml)',title='Water Awarded',
                marker='o',ax=ax,legend=False)
    #plot mean water reward data.
    plot_data.plot(x='session',y='reward',color='blue',
                xlabel='Session (#)',ylabel=u'Water per Reward (\u03bcl)',
                marker='o',ax=ax2,legend=False)

def table_period_info(ax,vr):
    period = vr.period[['period','set','duration','is_guided']].copy()
    # set NA of set to ''
    period.set = period.set.astype('string')
    period.loc[period.set=='NA','set']=''
    # set duration to mins float.
    period.duration = period.duration.dt.seconds/60
    # format
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    # rename columns
    period = period.rename(columns={'duration':'Duration (min)','period':'Type',
                                'set':'Cue Set','is_guided':'Is Guided'})
    table = ax.table(cellText=period.values, colLabels=period.keys(), loc='top',cellLoc='center',
            colColours=['lightblue','lightblue','lightblue','lightblue'],
            colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)   
    # adjust height header.
    for icol in range(period.shape[1]):
        cell = table[0, icol]
        cell.set_height(0.1)
        cell.set_text_props(fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def plot_num_trials_time(ax,vr):
    trials = vr.trial.copy()
    trials = trials.loc[trials.status!='INCOMPLETE']
    trials['num_trials'] = range(trials.shape[0])
    trials['time'] = trials['time_end']/ np.timedelta64(1, 'm')
    # plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    trials.plot(x='time',y='num_trials',xlabel = 'Time (min)',ylabel='Total Trials',
                ax=ax,legend=False,color='black')

def plot_total_water_time(ax,vr):
    water = vr.reward.copy()
    water['total_water'] = water.amount.cumsum()/1000
    water['time'] = water['time']/ np.timedelta64(1, 'm')
    # plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    water.plot(x='time',y='total_water',xlabel = 'Time (min)',ylabel='Total water (ml)',
                ax=ax,legend=False,color='black')

def plot_lick_delay_time(ax,vr):
    reward = vr.reward.copy()
    # find second closest lick for each reward (in case water drop is detected)
    ind = vr.lick['time'].searchsorted(reward['time'])+1
    ind[ind>= vr.lick.shape[0]] = 0
    reward['delta_lick_time'] = vr.lick['time'].iloc[ind].values - reward['time']
    reward['delta_lick_time'] = reward['delta_lick_time'].mask(reward['delta_lick_time']<np.timedelta64(0, 'm'))
    # change units
    reward['time'] = reward['time']/ np.timedelta64(1, 'm')
    reward['delta_lick_time'] = reward['delta_lick_time']/ np.timedelta64(1, 's')
    # plot found licks.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    reward.plot(x='time',y='delta_lick_time',xlabel = 'Time (min)',ylabel='Lick Delay (sec)',
                ax=ax,legend=False,color='green',label='With Lick')
    # 
    reward['fill_value'] = 0
    reward.loc[reward['delta_lick_time'].isnull()].plot(x='time',y='fill_value',xlabel = 'Time (min)',ylabel='Lick Delay (sec)',
                ax=ax,legend=False,color='red',label='No Lick')
    #ax.set_yscale('log')
    ax.legend(loc='upper right')

def table_session_settings(ax,vr):
    settings = vr.settings.copy()
    # select fields.
    settings = settings[['time','teleport_dark_duration','incorrect_dark_duration','teleport_fade_duration',
                        'poisson_lambda','max_repeats','freeze_on_incorrect','allow_multiple_responses']]
    # change to seconds.
    settings['poisson_lambda'] = settings['poisson_lambda'].round(2)
    # change to elapsed minutes.
    settings['time'] = settings['time']/ np.timedelta64(1, 'm')
    settings.time = settings.time.round(2)
    # remove none changed settings rows.
    check_fields = ['teleport_dark_duration','incorrect_dark_duration','poisson_lambda','max_repeats','teleport_fade_duration','freeze_on_incorrect','allow_multiple_responses']
    settings['inc'] = settings.groupby(check_fields).diff().fillna(True)
    settings = settings.loc[settings.inc==True].drop(columns='inc')
    # change column names. 
    settings = settings.rename(columns={'time':"Time\n(min)"
                                        ,'teleport_dark_duration':"Teleport Dark\nDuration (sec)",
                            'incorrect_dark_duration':"Incorrect Dark\nDuration (sec)",
                                        'teleport_fade_duration':'Teleport Fade\nDuration (sec)',
                                    'poisson_lambda':'Poisson Lambda',
                                    'max_repeats':'Max. Repeats',
                                    'freeze_on_incorrect':'Freeze\nIncorrect',
                                    'allow_multiple_responses':'Multiple\nResponses'})
    # plot.
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    table=ax.table(cellText=settings.values, colLabels=settings.keys(), loc='top',cellLoc='center',
            colColours=['lightblue' for i in range(settings.shape[1])],
            colWidths=np.array([0.2, 0.25, 0.25,0.25,0.25,0.25,0.25,0.25])/1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # adjust height header.
    for icol in range(settings.shape[1]):
        cell = table[0, icol]
        cell.set_height(0.175)
        cell.set_fontsize(7)
        cell.set_text_props(fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def scatter_plot_trial_responses(ax,vr,cue_set):
    # get status and reward arm.
    trials = vr.trial.loc[(vr.trial['set']==cue_set),['time_end','is_guided','status','reward_id']]
    # format.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (mins)')
    ax.set_ylabel('Reward Position (#)')
    ax.set_ylim(0.5,2.5)
    ax.set_yticks([1,2])
    # scatter groups.
    selected = trials.loc[(trials['status']=='CORRECT') & (trials['is_guided']==False)]
    ax.scatter(x=selected['time_end'].dt.total_seconds()/60,y=selected['reward_id'],color='green',s=800,marker='|',label='Correct')
    selected = trials.loc[(trials['status']=='INCORRECT') & (trials['is_guided']==False)]
    ax.scatter(x=selected['time_end'].dt.total_seconds()/60,y=selected['reward_id'],color='red',s=800,marker='|',label='Incorrect')
    selected = trials.loc[(trials['status']=='CORRECT') & (trials['is_guided']==True)]
    ax.scatter(x=selected['time_end'].dt.total_seconds()/60,y=selected['reward_id'],color='green',s=100,marker='|')
    selected = trials.loc[(trials['status']=='NO_RESPONSE') & (trials['is_guided']==True)]
    ax.scatter(x=selected['time_end'].dt.total_seconds()/60,y=selected['reward_id'],color='blue',s=100,marker='|')
    selected = trials.loc[(trials['status']=='NO_RESPONSE') & (trials['is_guided']==False)]
    ax.scatter(x=selected['time_end'].dt.total_seconds()/60,y=selected['reward_id'],color='blue',s=800,marker='|',label='No Response')

def plot_rolling_trial_responses(ax,vr,cue_set,clip=False):
    # find trials
    trial_numbers = vr.trial.loc[(vr.trial['set']==cue_set) &
                                (vr.trial['is_guided']==False) &
                                (vr.trial['status'].isin(['CORRECT','INCORRECT','NO_RESPONSE'])),'trial_number']
    trials = vr.trial
    trials = trials.loc[trials['trial_number'].isin(trial_numbers)].copy()
    # rolling average accuracy.
    win_size=10
    trials['correct'] = trials['status']=='CORRECT'
    trials['incorrect'] = trials['status']=='INCORRECT'
    trials['no_response'] = trials['status']=='NO_RESPONSE'
    trials['count'] = trials.rolling(win_size,on='trial_number',min_periods=1,center=True)['status'].count()
    trials['correct'] = (trials.rolling(win_size,on='trial_number',min_periods=1,center=True)['correct'].sum()/trials['count'])*100
    trials['incorrect'] = (trials.rolling(win_size,on='trial_number',min_periods=1,center=True)['incorrect'].sum()/trials['count'])*100
    trials['no_response'] = (trials.rolling(win_size,on='trial_number',min_periods=1,center=True)['no_response'].sum()/trials['count'])*100
    # fill in missing values.
    new_index = pd.Index(np.arange(trials['trial_number'].min(),trials['trial_number'].max()+1,1), name="trial_number")
    trials = trials.set_index("trial_number").reindex(new_index).reset_index()
    # plot.
    ax.plot(trials['trial_number'],trials['correct'],color='g' , clip_on=clip,label='Correct')
    ax.plot(trials['trial_number'],trials['incorrect'],color='r' , clip_on=clip,label='Incorrect')
    ax.plot(trials['trial_number'],trials['no_response'],color='b' , clip_on=clip,label='No Response')
    # format.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Trial (#)')
    ax.set_ylabel(f'Occurence(%)\nRolling Average of {win_size}')
    ax.set_ylim(0,100)
    ax.legend(loc='upper right')

def plot_bar_response_per_reward_pos(ax,vr,cue_set,rewarding_id):
    # find trials with set.
    trials = vr.trial.loc[(vr.trial.set==cue_set) & (vr.trial.reward_id==rewarding_id)& 
                        (vr.trial.status!='INCOMPLETE') & (vr.trial['is_guided']==False)]
    # get counts.
    trials = trials.groupby('status')['trial_number'].agg('count')
    trials = trials.drop('INCOMPLETE')
    trials = (trials/trials.sum())*100
    # format axis.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    trials.plot.barh(ax=ax,y='status',color=['green','red','blue'],xlabel='',title=f'Reward Position #{rewarding_id}',edgecolor='black')
    ax.set_xlabel('Occurence (%)')
    ax.set_xlim([0,100])
    ax.invert_yaxis()

def scatter_licks_per_condition(ax,vr,cue_set,rewarding_id):
    # show cue areas.
    num_reward_pos = len(vr.environment.reward_positions[0])
    for i in range(num_reward_pos):
        color_cue = 'r'
        if (i+1)==rewarding_id:
            color_cue= 'g'
        min_x = vr.environment.reward_positions[0][i]-(vr.environment.reward_sizes[0][i]/2)
        max_x = vr.environment.reward_positions[0][i]+(vr.environment.reward_sizes[0][i]/2)
        ax.axvspan(min_x, max_x, color=color_cue, alpha=0.3,linewidth=0)
    # show indicator
    min_x = (vr.environment.indicator_position.values-(vr.environment.indicator_size.values/2))[0]
    max_x = (vr.environment.indicator_position.values+(vr.environment.indicator_size.values/2))[0]
    ax.axvspan(min_x, max_x, color='b', alpha=0.3,linewidth=0)
    # show gray zone.
    min_x = (vr.environment.gray_zone_position.values-(vr.environment.gray_zone_size.values/2))[0]
    max_x = (vr.environment.gray_zone_position.values+(vr.environment.gray_zone_size.values/2))[0]
    # plot end
    ax.axvline( x=vr.environment['stop_position'].values,color='red',linestyle='--')
    ax.axvspan(min_x, max_x, color='gray', alpha=0.3,linewidth=0)

    # find trials with set.
    # non guided
    trials = vr.trial.loc[(vr.trial.set==cue_set) & (vr.trial.reward_id==rewarding_id) & (vr.trial.is_guided==False)
        & (vr.trial.status!='INCOMPLETE'),'trial_number']
    licks = vr.lick.loc[vr.lick.trial_number.isin(trials)]
    licks.plot.scatter(x='position',y='trial_number',ax=ax,color='blue',edgecolor='black',title=f'Reward Position #{rewarding_id}')
    # guided
    trials = vr.trial.loc[(vr.trial.set==cue_set) & (vr.trial.reward_id==rewarding_id) & (vr.trial.is_guided)
        & (vr.trial.status!='INCOMPLETE'),'trial_number']
    licks = vr.lick.loc[vr.lick.trial_number.isin(trials)]
    licks.plot.scatter(x='position',y='trial_number',ax=ax,color='green',edgecolor=None,title=f'Reward Position #{rewarding_id}')
    #format axis.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Trial (#)')
    ax.set_xlim(0,vr.environment['stop_position'].values)
    ax.invert_yaxis()

def bar_licks_position_area(ax,vr,cue_set,rewarding_id):
    # find trials with set.
    trials = vr.trial.loc[(vr.trial.set==cue_set) & (vr.trial.reward_id==rewarding_id)
        & (vr.trial.status!='INCOMPLETE') & (vr.trial.is_guided==False),'trial_number']
    # get licks info.
    licks = vr.lick.loc[vr.lick.trial_number.isin(trials)]
    # group by trial.
    min_rew1 = (vr.environment.reward_position_1-(vr.environment.reward_size_1/2)).values[0]
    max_rew1 = (vr.environment.reward_position_1+(vr.environment.reward_size_1/2)).values[0]
    min_rew2 = (vr.environment.reward_position_2-(vr.environment.reward_size_2/2)).values[0]
    max_rew2 = (vr.environment.reward_position_2+(vr.environment.reward_size_2/2)).values[0]
    licks = licks.groupby('trial_number').agg(
    rew1=('position',lambda x: sum((x>=min_rew1) & (x<=max_rew1))),
    rew2=('position',lambda x: sum((x>=min_rew2) & (x<=max_rew2))),
    total=('position','count')
    )
    licks['outside'] = licks['total']-(licks['rew1']+licks['rew2'])
    licks.rew1 = (licks.rew1/licks.total)*100
    licks.rew2 = (licks.rew2/licks.total)*100
    licks.outside = (licks.outside/licks.total)*100
    ## plot.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(f'Reward Position #{rewarding_id}')
    ax.set_xlabel('Licks (%)')
    ax.set_xlim(0,100)
    ax.invert_yaxis()
    # calc stats (avoids warnings due to NaN).
    stats = []
    for field in ['rew1','rew2','outside']:
        m = licks[field].mean()
        if np.isnan(m):
            m=0
        sem = licks[field].sem()
        if np.isnan(sem):
            sem =0
        stats.append({'mean':m,'sem':sem})
    # bar plot.
    bars = ax.barh(['Reward #1', 'Reward #2', 'Outside'],[stats[0]['mean'],stats[1]['mean'],stats[2]['mean']],
        xerr = [(0,0,0), [stats[0]['sem'],stats[1]['sem'],stats[2]['sem']]], capsize=4,edgecolor='black', color='red')
    bars.patches[rewarding_id-1].set_color('green')
    bars.patches[rewarding_id-1].set_edgecolor('black')
    bars.patches[2].set_color('gray')
    bars.patches[2].set_edgecolor('black')