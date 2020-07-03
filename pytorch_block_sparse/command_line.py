import click
from . import run1, run2

@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj = {}

@cli.command()
@click.pass_context
@click.argument('path', default = None, type=click.Path(exists = True, resolve_path = True))
@click.argument('output', default = None, type=click.Path(resolve_path = True))
@click.option('--arg', '-a', is_flag = True)
def command1(ctx, path, output, arg):
    click.echo(click.style("Hello world !", fg="red"))
    print(path + ":" + run1(path, output))

@cli.command()
@click.pass_context
@click.argument('path', default = None, type = click.Path(exists = True, resolve_path = True))
@click.argument('output', default = None, type = click.Path(resolve_path = True))
def command2(ctx, path, output):
    print(path + ":" + run2(path, output))

def main():
    return cli()