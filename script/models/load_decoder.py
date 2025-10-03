from script.models.Decoders.model import NCModel, LPModel


def load_decoder(args, logger):
    if args.task == 'nc':
        decoder = NCModel(args)
    elif args.task == 'lp':
        decoder = LPModel(args)
    else:
        raise Exception('pls define the decoder for {} task'.format(args.task))
    logger.info('using decoder in {} task'.format(args.task))
    return decoder