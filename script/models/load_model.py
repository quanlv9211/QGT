from script.models.QGCN2.model import QGCN2
from script.models.QGT.model import QGT


def load_model(args, logger):
    if args.model == 'QGCN2':
        model = QGCN2(args)
    elif args.model == 'QGT':
        model = QGT(args)
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.model))
    return model
