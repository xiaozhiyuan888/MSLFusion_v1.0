import os
from PIL import Image
from tqdm import tqdm


def tif_to_jpg(input_path, output_path, quality=95, overwrite=False):
    """
    Convert a single TIF file to a JPG file

    Parameters:
    input_path (str): Path to the input TIF file
    output_path (str): Path to the output JPG file
    quality (int): JPG image quality, ranging from 0 to 100, default 95
    overwrite (bool): Whether to overwrite existing files, default False
    """

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path) and not overwrite:
        print(f"Skip existing files: {output_path}")
        return

    try:
        with Image.open(input_path) as img:
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')

            img.save(output_path, 'JPEG', quality=quality)
            print(f"Successfully converted: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Conversion failed: {input_path}, error: {str(e)}")


def batch_convert(input_dir, output_dir, quality=95, overwrite=False, recursive=False):
    """
    Batch convert TIF files to JPG files

    Parameters:
    input_dir (str): Input directory path
    output_dir (str): Output directory path
    quality (int): JPG image quality, ranging from 0 to 100, default 95
    overwrite (bool): Whether to overwrite existing files, default False
    recursive (bool): Whether to process subdirectories recursively, default False
    """

    if not os.path.exists(input_dir):
        print(f"The input directory does not exist: {input_dir}")
        return

    tif_files = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    tif_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(input_dir, file))

    if not tif_files:
        print(f"in the directory {input_dir} No TIF files found")
        return

    print(f"found {len(tif_files)} TIF files found, starting conversion...")
    for tif_file in tqdm(tif_files):
        relative_path = os.path.relpath(tif_file, input_dir)
        output_subdir = os.path.dirname(relative_path)
        output_subdir_path = os.path.join(output_dir, output_subdir)

        base_name = os.path.basename(tif_file)
        file_name, _ = os.path.splitext(base_name)
        output_file_name = f"{file_name}.jpg"
        output_file_path = os.path.join(output_subdir_path, output_file_name)

        tif_to_jpg(tif_file, output_file_path, quality, overwrite)

    print(f"Conversion completed! All JPG files have been saved to: {output_dir}")


if __name__ == "__main__":
    input_path = r"Replace with your TIF file or directory path"
    output_path = r"Replace with your output directory path"

    quality = 95  # JPG quality, ranging from 0 to 100，default 95
    overwrite = False  # Whether to overwrite existing files，default False
    recursive = True  # Whether to process subdirectories recursively, default True

    if os.path.isfile(input_path):
        if os.path.isdir(output_path):
            base_name = os.path.basename(input_path)
            file_name, _ = os.path.splitext(base_name)
            output_path = os.path.join(output_path, f"{file_name}.jpg")

        tif_to_jpg(input_path, output_path, quality, overwrite)
    elif os.path.isdir(input_path):
        batch_convert(input_path, output_path, quality, overwrite, recursive)
    else:
        print(f"The input path does not exist: {input_path}")