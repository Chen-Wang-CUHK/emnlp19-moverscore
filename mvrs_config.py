import configargparse as cfargparse

ArgumentParser = cfargparse.ArgumentParser

def general_args(parser):
    """
    add some general options
    """
    group = parser.add_argument_group("general_args")
    group.add('--dataset', '-dataset', type=str, default='tac.09.mds.gen.resp-pyr',
              help="The input data file")