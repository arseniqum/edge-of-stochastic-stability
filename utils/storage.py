import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def get_welcome_string(args):
    """
    Returns the header of the log file.

    Parameters:
    -----------
    args : argparse object with the parameters given to the program
    """

    msg = f"""# Edge of Stochastic Stability.
# Dataset: {args.dataset}, model {args.model}, lr {args.lr}, batch size {args.batch}, gd_noise {args.gd_noise}
# Arguments: {str(args)}
# (0) epoch, (1) step, (2) batch loss, (3) full loss, (4) lambda max, (5) step sharpness, (6) batch sharpness, (7) Gradient-Noise Interaction, (8) total accuracy"""
    return msg


def initialize_folders(args, results_folder):
    FOLDER_ROOT_IN_RESULTS = 'plaintext'
    run_folder_name = f'{args.dataset}_{args.model}'

    def generate_folder_name(args):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M_%S')
        config_name = f'{timestamp}_lr{args.lr:.5f}_b{args.batch}' 
        return config_name
    
    while True:
        config_name = generate_folder_name(args)
        runs_folder = results_folder / FOLDER_ROOT_IN_RESULTS / run_folder_name / config_name
        if not runs_folder.exists():
            try:
                runs_folder.mkdir(parents=True, exist_ok=False)
            except:
                time.sleep(2)
                continue
            break
        else:
            time.sleep(2)
            continue



    model_save_path = runs_folder / 'checkpoints'
    model_save_path.mkdir(parents=True, exist_ok=True)


    results_file = runs_folder / 'results.txt'
    welcome_string = get_welcome_string(args)

    with open(results_file, 'w') as f:
        f.write(welcome_string + "\n")

    return runs_folder


def write_json_atomic(path: Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    """
    Persist a mapping as JSON using a temporary file to avoid partial writes.

    Parameters
    ----------
    path : Path
        Destination file path.
    payload : Mapping[str, Any]
        Data to serialise as JSON.
    indent : int, optional
        Indentation level for pretty-printing. Default is 2.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)
    tmp_path.replace(path)
