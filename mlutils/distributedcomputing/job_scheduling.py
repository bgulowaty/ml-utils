import subprocess


def run_experiments_in_slurm(run_ids, notebook_path, output_dir_path, papermill_path, script_path="slurm-script.sh",
                             notebook_run_id_param="EXPERIMENT_INSTANCE_ID"):
    futures = []

    for run_id in run_ids:
        # print(f"Running file {file_name}")
        papermill_command = f"{str(papermill_path)} {str(notebook_path)} {str(output_dir_path)}/{run_id}.ipynb -p {notebook_run_id_param} {run_id}"
        std_out_path = output_dir_path / f"{run_id}.out"
        std_err_path = output_dir_path / f"{run_id}.err"

        slurm_command = f"sbatch -o {str(std_out_path)} -e {str(std_err_path)} {script_path} \"{papermill_command}\""

        subprocess.run(slurm_command, shell=True)

    return futures


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


how_many_jobs = lambda username: int(
    subprocess.run(f"squeue | grep {username} | wc -l", stdout=subprocess.PIPE, shell=True).stdout.decode(
        'utf-8').strip())


def run_in_batches(run_ids, notebook_path, output_dir_path, username='bogul', batch_size=250, sleep_interval=25):
    chunked_run_ids = chunks(run_ids, batch_size)

    while (True):
        jobs_currently = how_many_jobs(username)
        if jobs_currently <= batch_size:
            print(f"There are {jobs_currently} runnig jobs, scheduling next batch of {batch_size}")
            try:
                next_batch = next(chunked_run_ids)
                run_experiments_in_slurm(next_batch, notebook_path, output_dir_path)
            except StopIteration:
                print("End of batches!")
                break
        else:
            print(f"There are {jobs_currently}, cant schedule yet!")
        print(f"Waiting {sleep_interval}")
        time.sleep(sleep_interval)
