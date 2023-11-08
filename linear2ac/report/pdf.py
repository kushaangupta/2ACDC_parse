from fpdf import FPDF
import figrid as fg
from pathlib import Path
import matplotlib.pyplot as plt
import uuid
import shutil
from linear2ac.io import collect_log_data
import linear2ac.report.plots as plots
from tqdm.notebook import tqdm

def generate_report(main_folder,force_reparse=False,verbose=True, custom_file_name=None):
    # get log data.
    vrs = collect_log_data(main_folder,force_reparse,verbose=verbose)
    if not vrs:
        raise NameError("Could not find any data folders in format: 2020_01_01/1")
    # initialize pdf.
    pdf = PDF_Template(Path(main_folder).name,main_folder,force_reparse)
    # add all sessions summary.
    all_sessions_summary(pdf,vrs)
    # add cue set session summary pages.
    for isession, vr in tqdm(reversed(list(enumerate(vrs))),desc = 'Creating session pages',disable=not verbose,total=len(vrs)):
        # check if licks were detected (in rare cases if there was an issue)
        if not vr.lick.empty:
            session_summary(pdf,vr,isession)
            # get cue sets in session.
            cue_sets = vr.trial.set.unique()
            for cue_set in cue_sets:
                cue_set_summary(pdf,vr,cue_set,isession)
    # output pdf.
    pdf.generate_pdf(custom_file_name)

def all_sessions_summary(pdf,vrs):
    label = 'all session_summary'
    pdf.add_page()
    pdf.section_title(f'All Sessions Summary',18)  
    fig = plt.figure(figsize=(210/25.4,270/25.4),edgecolor='black')
    ax = {
    'max performance sessions': fg.place_axes_on_grid(fig, xspan=[0, 0.95], yspan=[0.0, 0.25]),
    'performance all sessions': fg.place_axes_on_grid(fig, xspan=[0, 0.95], yspan=[0.35, 0.6]),
    'trials total': fg.place_axes_on_grid(fig, xspan=[0, 0.45], yspan=[0.7, 0.95]),
    'water total': fg.place_axes_on_grid(fig, xspan=[0.55, 1], yspan=[0.7, 0.95]),
    }
    plots.scatter_summary_performance(ax['performance all sessions'],vrs)
    plots.scatter_summary_max_performance(ax['max performance sessions'],vrs)
    plots.plot_summary_total_trials(ax['trials total'],vrs)
    plots.plot_summary_total_water(ax['water total'],vrs)
    pdf.add_figure_to_page(fig,label)

def session_summary(pdf,vr,isession):
    label = 'session_summary'
    pdf.add_page()
    pdf.section_title(f'Session {isession+1}: {vr.info["date_time"]}',18)  
    if (pdf.force_recalc) | (pdf.add_image_to_page(label,vr)==False):
        fig = plt.figure(figsize=(210/25.4,270/25.4),edgecolor='black')
        ax = {
            'session_periods': fg.place_axes_on_grid(fig, xspan=[0, 0.95], yspan=[0.05, 0.35]),
            'trial total': fg.place_axes_on_grid(fig, xspan=[0, 0.45], yspan=[0.15, 0.35]),
            'water total': fg.place_axes_on_grid(fig, xspan=[0.6, 1], yspan=[0.15, 0.35]),
            'lick delay': fg.place_axes_on_grid(fig, xspan=[0, 0.45], yspan=[0.425, 0.625]),
            'settings': fg.place_axes_on_grid(fig, xspan=[0, 0.95], yspan=[0.75, 0.95]),
        }
        plots.table_period_info(ax['session_periods'],vr)
        plots.plot_num_trials_time(ax['trial total'],vr)
        plots.plot_total_water_time(ax['water total'],vr)
        plots.plot_lick_delay_time(ax['lick delay'],vr)
        plots.table_session_settings(ax['settings'],vr)
        pdf.add_figure_to_page(fig,label,vr)

def cue_set_summary(pdf,vr,cue_set,isession):
    label = f'cue_set_summary - {cue_set}'
    pdf.add_page()
    pdf.section_title(f'Session {isession+1}: {vr.info["date_time"]}',18)
    pdf.section_title(f'{cue_set}',16)
    # test if image is already generated or if there is a forced recalculation.
    if (pdf.force_recalc) | (pdf.add_image_to_page(label,vr)==False):
        # generate figure.
        fig = plt.figure(figsize=(210/25.4,270/25.4),edgecolor='black')
        ax = {
        'scatter_plot_trial_responses': fg.place_axes_on_grid(fig, xspan=[0, 1], yspan=[0, 0.2]),
        'plot rolling trial responses': fg.place_axes_on_grid(fig,xspan=[0, 0.55],yspan=[0.275,0.5]),
        'plot bar response reward pos 1': fg.place_axes_on_grid(fig,xspan=[0.75, 1],yspan=[0.3,0.35]),
        'plot bar response reward pos 2': fg.place_axes_on_grid(fig,xspan=[0.75, 1],yspan=[0.45,0.5]),
        'scatter licks per condition 1': fg.place_axes_on_grid(fig,xspan=[0, 0.3],yspan=[0.6,0.975]),
        'scatter licks per condition 2': fg.place_axes_on_grid(fig,xspan=[0.4, 0.7],yspan=[0.6,0.975]),
        'bar licks position area 1': fg.place_axes_on_grid(fig,xspan=[0.85, 1],yspan=[0.6,0.74]),
        'bar licks position area 2': fg.place_axes_on_grid(fig,xspan=[0.85, 1],yspan=[0.835,0.975]),
        }
        plots.scatter_plot_trial_responses(ax['scatter_plot_trial_responses'],vr,cue_set)
        plots.plot_rolling_trial_responses(ax['plot rolling trial responses'],vr,cue_set)
        plots.plot_bar_response_per_reward_pos(ax['plot bar response reward pos 1'],vr,cue_set,1)
        plots.plot_bar_response_per_reward_pos(ax['plot bar response reward pos 2'],vr,cue_set,2)
        plots.scatter_licks_per_condition(ax['scatter licks per condition 1'],vr,cue_set,1)
        plots.scatter_licks_per_condition(ax['scatter licks per condition 2'],vr,cue_set,2)
        plots.bar_licks_position_area(ax['bar licks position area 1'],vr, cue_set,1)
        plots.bar_licks_position_area(ax['bar licks position area 2'],vr, cue_set,2)
        pdf.add_figure_to_page(fig,label,vr)

class PDF_Template(FPDF):
    """Template for pdf page with footer header and page specifications.

    Args:
        FPDF (FPDF): This is the class constructor. It allows setting up the page format, the orientation and the unit of measurement used in all methods (except for font sizes).
    """
    animal = ''
    main_folder ='' # main root folder
    temp_folder = '' # temporary image file folder.
    force_recalc = False # force caculation of figures.
    def __init__(self,animal,main_folder,force_recalc=False):
        super(PDF_Template,self).__init__('P','mm','Letter')
        self.animal = animal
        self.main_folder = Path(main_folder)
        self.temp_folder = Path(self.main_folder)/'tmp'
        self.alias_nb_pages('{nb}')
        self.force_recalc = force_recalc
    def add_image_to_page(self,label,vr=None):
        # find imaging.
        file_name = self.get_image_file_name(label,vr)
        if file_name.is_file():
            self.image(file_name,x = 5, y=20, w=210,h=270)
            return True
        else:
            return False

    def get_image_file_name(self,label,vr=None):
        if not vr:
            prefix=''
        else:
            prefix = vr.info['date_time'].replace(':','-')
        return self.temp_folder/f'{prefix} {label}.png'

    def add_figure_to_page(self,fig,label,vr=None):
        """Takes figure generating function and adds it to the page.

        Args:
            fnc (Function handle): Must have outputs figure handle, and axis handle
        """
        file_name = self.get_image_file_name(label,vr)
        self.save_fig_as_png(fig,file_name)
        self.image(file_name,x = 5, y=20, w=210,h=270)

    def clean_up_temp_folder(self):
        """Remove temporary image folder
        """
        shutil.rmtree(self.temp_folder)
    def save_fig_as_png(self,fig,file_name):
        """Saves figure as png in temporary folder within the main folder.

        Args:
            main_folder (string): Data folder root.
            fig (matplotlib figure handle): Figure to be saved

        Returns:
            string: location of saved figure png
        """
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_name,dpi=200, transparent=True)
        plt.close(fig)
        return file_name
    
    def generate_pdf(self, custom_file_name):
        """Wrapper function for generating report pdf.
        """
        # output report
        if custom_file_name:
            self.output(self.main_folder/f'{custom_file_name}.pdf')
        else:
            self.output(self.main_folder/f'{self.animal} report.pdf')
        # clean up temporary files.
        #self.clean_up_temp_folder()

    def header(self):
        # Logo
        self.line(10, 22, 205, 22)
        self.image(Path(__file__).parent/'assets'/'HHMI_Janelia_Color.png', 10, 8, 33)
        # helvetica bold 15
        self.set_font('helvetica', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(50, 10, f'Behavior Report - {self.animal}', 0, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # helvetica italic 8
        self.set_font('helvetica', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        
    def section_title(self, label, fontsize):
        # helvetica 12
        self.set_font('helvetica', 'b', fontsize)
        # Title
        self.cell(0, 3, f'{label}', 0, 1, 'L', False)
        self.set_font('helvetica', '', 10)
        # Line break
        self.ln(6)