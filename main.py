from lib import extraction as gen, constants
import click
import warnings

@click.group()
def main():
	pass

# Used to segment all images in Syntheyes directory
# :arg1: input- directory path to Syntheyes Dataset
# :arg2: output- output directory where all segmented images will be saved
@main.command()
@click.argument("input")
@click.argument("output")
def dir(input, output):
	gen.segment_folder(input, output)


# Template argument for converting a image
@main.command()
@click.argument("input")
@click.argument("output")
def img(input, output):
	click.echo(input)


if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	main()