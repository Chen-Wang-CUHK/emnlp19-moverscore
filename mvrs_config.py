import configargparse as cfargparse

ArgumentParser = cfargparse.ArgumentParser

def general_args(parser):
    """
    add some general options
    """
    group = parser.add_argument_group("general_args")
    group.add('--dataset', '-dataset', type=str, default='tac.09.mds.gen.resp-pyr',
              help="The input data file")
    group.add('--mvrs_type', '-mvrs_type', type=str, default='v2', choices=['v1', 'v2'],
              help="The moverscore version. v1 uses mnli-bert model and v2 uses distilbert-base-uncased model")