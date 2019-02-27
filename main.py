from lib import extraction as gen, constants
import click
import warnings

@click.group()
def main():
	pass
"""
Used to segment all images in Syntheyes directory
:arg1: input_directory- directory path to Syntheyes Dataset
:arg2: output_directory- output directory where all segmented 
       images will be saved
"""
@main.command()
@click.argument("input_directory")
@click.argument("output_directory")
def dir(input_directory, output_directory):
	gen.segment_folder(input_directory, output_directory)

"""
 Generate Iris mask from landmarks
 :arg1: input_file_path- path to pickle 
 :arg2: output_directory- output directory
"""
@main.command()
@click.argument("input_file_path")
@click.argument("output_directory")
def img(input_file_path, output_directory):
	gen.segment_from_pickle(input_file_path, output_directory)

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	main()