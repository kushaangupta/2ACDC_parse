from pathlib import Path
import zarr
import vr2p
import linear2ac.place
import numcodecs
import vr2p
from pathlib import Path
import uuid
from suite2p.io.server import ssh_connect
import re
import yaml
from tqdm.contrib.concurrent import process_map

def process_placefield_data(input_path, output_path, settings_file, session_id, cueset, reward_id, trial_status,bin_size=5,force_recalc=False):
    """_summary_

    Args:
        input_path (_type_): _description_
        output_path (_type_): 
        session_id (_type_): _description_
        cueset (_type_): _description_
        reward_id (_type_): _description_
        trial_status (_type_): _description_
        bin_size (int, optional): _description_. Defaults to 5.
        force_recalc (bool, optional): _description_. Defaults to False.

    Raises:
        NameError: _description_
        NameError: _description_
        NameError: _description_
    """
    input_path = Path(input_path)
    output_path = Path(output_path) # relative to input path so its easier to check reuslts on windows system.
    # debug messages.
    print(f'input path: {input_path}')
    print(f'output path: {output_path}')
    print(f'session id: {session_id}')
    print(f'cueset: {cueset}')
    print(f'reward id: {reward_id}')
    print(f'trial_status: {trial_status}')
    # read settings file.
    with open(settings_file, "r") as stream:
        settings = yaml.safe_load(stream)
    # open input zarr folder.
    if not input_path.is_dir():
        raise NameError(f'Could not find folder {input_path}')
    data = vr2p.ExperimentData(input_path)
    print('Loaded data')
    # open output zarr folder.
    if not output_path.parent.is_dir():
        raise NameError(f'Could not find folder {output_path.parent}')
    with zarr.open(output_path.as_posix(), mode="a") as zarr_output:
        # check if data is already present.
        data_path = f"{cueset}/{reward_id}/{trial_status}/{session_id}"
        print(f"zarr data path: '{data_path}'")
        if (f"{data_path}/significant" in zarr_output) & (force_recalc==False):
            raise NameError(f'Data already present and force_recalc was set to False')
        if (f"{data_path}/significant" in zarr_output) & (force_recalc):
            print(f'Removing existing data in {data_path}')
            zarr_output[data_path].clear()
        # create group.
        zarr_dataset = zarr_output.create_group(data_path,overwrite=True)
        
        # calculate place field info.
        vr = data.vr[int(session_id)]
        trial = vr.trial.copy()
        position = vr.path.frame.copy().reset_index()
        # select trials.
        if trial_status=='all':
            selected_trials = trial.loc[(trial.reward_id == reward_id) 
                                        & (trial.set==cueset),'trial_number']
        if trial_status=='correct':
            selected_trials = trial.loc[(trial.reward_id == reward_id) 
                                        & (trial.set==cueset)
                                        & (trial.status=='CORRECT')
                                        & (trial.is_guided=='False'),'trial_number']
        if trial_status=='incorrect':
            selected_trials = trial.loc[(trial.reward_id == reward_id) 
                                        & (trial.set==cueset)
                                        & (trial.status=='INCORRECT')
                                        & (trial.is_guided=='False'),'trial_number']
        if trial_status=='excl_no_response':
            selected_trials = trial.loc[(trial.reward_id == reward_id) 
                                        & (trial.set==cueset)
                                        & (trial.status!='NO_RESPONSE'),'trial_number']
        # select_frames
        selected_frames = position.loc[position['trial_number'].isin(selected_trials),'frame']
        # Detect place cells.
        pf_detect = linear2ac.place.Tyche1dProtocol()
        # set config values from settings file.
        for key in settings['config'].keys():
            setattr(pf_detect.config,key,settings['config'][key])
        # run pf detection.
        pf_detect.detect(data.signals.multi_session.Fdemix[int(session_id)], vr, bin_size, selected_frames,
         verbose=True, use_parallel=settings['server']['use_parallel'],
         parallel_processes = settings['server']['parallel_processes'])
        # store data
        print('Storing results..')
        def store_data(dataset_name, field_props,pf):
            pf_data = {'field_props':field_props,'binF':pf.binF, 'label_im':pf.label_im,'order':pf.order,
            'centers':pf.centers,'has_place_field':pf.has_place_field,'is_significant': pf_detect.is_significant}
            zarr_dataset.create_dataset(dataset_name, data= pf_data, dtype=object, object_codec = numcodecs.Pickle())
        # putative place fields (before checks).
        all_field_props = pf_detect.putative_field_props
        store_data('putative', all_field_props, pf_detect.pf_putative)
        store_data('passed', [prop for prop in all_field_props if prop['passed']], pf_detect.pf_passed)
        store_data('significant', [prop for prop in all_field_props if prop['passed'] & pf_detect.is_significant[prop['cell']]], pf_detect.pf_significant)     
        # store object.
        pf_detect.clean_up()
        zarr_dataset.create_dataset('pf', data= pf_detect, dtype=object, object_codec = numcodecs.Pickle())
        # store num_trials*num_bins*num_cell overview data
        def store_array(data_set_name, array, data_type):
            zarr_dataset.create_dataset(data_set_name, data = array, shape=array.shape, chunks=(100,100,50), dtype=data_type)
        store_array('binF_mat',pf_detect.binF_mat,'f')
        store_array('event_mat',pf_detect.event_mask_mat,'b')
        # store config.
        zarr_dataset.create_dataset('settings', data=settings, dtype=object, object_codec = numcodecs.Pickle())
        print('Done!')
        return pf_detect

def send_placefield_job(info, server):
    job_id = f"{info['session_id']}-{info['reward_id']}-{uuid.uuid4().hex[:3].upper()}"
    # connect to ssh
    ssh = ssh_connect(server['host'], server['username'], server['password'],verbose=False)
    # run command
    run_command = f"bsub -n {server['n_cores']} -J {job_id} "
    run_command +=f'-R"select[avx512]" -o logs/out-{job_id}.txt "~/placefield_job.sh'
    # arguments.
    for key in ['input_path','output_path','settings_file','session_id','cueset','reward_id','trial_status','bin_size','force_recalc']:
        run_command+= f' \'{info[key]}\''
    run_command+= f'> logs/pf-log-{job_id}.txt"'
    stdin, stdout, stderr = ssh.exec_command(run_command)
    # find job id.
    stdout = stdout.read().decode('utf-8')
    stdout = re.search('Job <(\d*)>',stdout)
    if stdout:
        job_id = int(stdout.group(1))
        return {'info':info,'job_id':job_id}
    else:
        raise NameError("Could not find job id (was job submit succesfull?)")
    pass

def check_placefield_job_status(info,local_output_path, server, verbose=True):
    """Checks on status of job generated by send_placefield_job.

    Args:
        info (dictionary): created by send_placefield_job
        server (dictionary): Necessary to connect to cluster.
                        'host'
                        'username'
                        'password'
    """
    ssh = ssh_connect(server['host'], server['username'], server['password'],verbose=False)
    # check job status
    stdin, stdout, stderr = ssh.exec_command(f'bjobs -l { info["job_id"] }')
    status = re.search("Status <.*?>",stdout.read().decode('utf-8')).group()
    # check result presence
    data_path = f"{info['info']['cueset']}/{info['info']['reward_id']}/{info['info']['trial_status']}/{info['info']['session_id']}"
    if check_placefield_results(data_path,local_output_path):
            result_folder = 'present'
    else:
        result_folder = 'absent'
    if verbose:
        print(f"{data_path:<35} {status:<10}, results: {result_folder:<10}")
    return re.search('Status <(.+?)>', status).group(1), result_folder

def check_placefield_results(data_path,local_output_path):
    """Check for presence of placefield analysis results

    Args:
        data_path (string): location in zarr object.
        local_output_path (string): location of zarr object.

    Returns:
        bool: True if "signficant" data array is present (created last)
    """
    # check combined folder 
    if not local_output_path.is_dir(): 
        return False
    with zarr.open(local_output_path.as_posix(), mode="r") as zarr_output:
        return (f"{data_path}/significant" in zarr_output)

def run_placefield_cluster(local_input_path, server_input_path, local_output_path, server_output_path, server_settings_file, settings, force_recalc):
    data = vr2p.ExperimentData(local_input_path)
    jobs = []
    for session_id, vr in enumerate(data.vr):
        for cueset in vr.trial.set.unique():
            for reward_id in [1,2]:
                for trial_status in ['correct','incorrect','excl_no_response','all']:
                    # gather info.
                    process_info = {'input_path':server_input_path.as_posix(),'output_path':server_output_path.as_posix(),
                    'settings_file': server_settings_file.as_posix(),
                    'session_id':session_id,'cueset':cueset,'reward_id':reward_id,
                    'trial_status':trial_status,'bin_size':settings['placefield']['bin_size'],
                    'force_recalc':force_recalc}
                    # check if results are already present.
                    data_path = f"{cueset}/{reward_id}/{trial_status}/{session_id}"
                    if (not check_placefield_results(data_path,local_output_path)) | (force_recalc):
                        jobs.append(send_placefield_job(process_info, settings['server']))
                        job_str = "Submitted"
                    else:
                        job_str = "Skipped"
                    print (f"{job_str:<10} - {data_path:<30}")
    return jobs