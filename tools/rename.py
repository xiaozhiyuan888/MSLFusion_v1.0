# This file can help you rename the files in the dataset to the format of "imagex_x".

import os
import re
import shutil
from typing import Dict, List, Optional, Tuple


def rename_files(source_dir: str = '.', target_dir: str = None, dry_run: bool = True) -> None:
    """
        Rename IR and VIS files according to rules

        Parameters:
        source_dir: Path of the source directory
        target_dir: Path of the target directory; if None, rename in the original directory
        dry_run: Whether to only display operations without executing them
    """
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = os.listdir(source_dir)

    file_pairs: Dict[int, Dict[str, str]] = {}

    ir_pattern = re.compile(r'IR(\d+)')
    vis_pattern = re.compile(r'VIS(\d+)')

    for filename in files:
        ir_match = ir_pattern.search(filename)
        vis_match = vis_pattern.search(filename)

        if ir_match:
            number = int(ir_match.group(1))
            if number not in file_pairs:
                file_pairs[number] = {'IR': None, 'VIS': None}
            file_pairs[number]['IR'] = filename
        elif vis_match:
            number = int(vis_match.group(1))
            if number not in file_pairs:
                file_pairs[number] = {'IR': None, 'VIS': None}
            file_pairs[number]['VIS'] = filename

    sorted_pairs = sorted(file_pairs.items(), key=lambda x: x[0])

    for new_index, (original_number, pair) in enumerate(sorted_pairs, 1):
        ir_file = pair.get('IR')
        vis_file = pair.get('VIS')

        if ir_file:
            ir_ext = os.path.splitext(ir_file)[1]
            new_ir_name = f"image{new_index}_1{ir_ext}"
            perform_rename(ir_file, new_ir_name, source_dir, target_dir, dry_run)

        if vis_file:
            vis_ext = os.path.splitext(vis_file)[1]
            new_vis_name = f"image{new_index}_2{vis_ext}"
            perform_rename(vis_file, new_vis_name, source_dir, target_dir, dry_run)


def perform_rename(old_name: str, new_name: str, source_dir: str, target_dir: Optional[str], dry_run: bool) -> None:
    source_path = os.path.join(source_dir, old_name)
    if target_dir:
        target_path = os.path.join(target_dir, new_name)
    else:
        target_path = os.path.join(source_dir, new_name)

    if dry_run:
        print(f"Rename {source_path} to {target_path}")
    else:
        try:
            if target_dir:
                shutil.copy2(source_path, target_path)
            else:
                os.rename(source_path, target_path)
            print(f"Successfully renamed {old_name} to {new_name}")
        except Exception as e:
            print(f"Failed to rename {old_name} to {new_name}: {e}")


if __name__ == "__main__":
     rename_files(source_dir='your path', target_dir=None, dry_run=False)