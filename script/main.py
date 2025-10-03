import os
import sys
import time
import warnings
import numpy as np
from math import isnan
import torch
import matplotlib.pyplot as plt

#torch.autograd.set_detect_anomaly(True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def run(args):
    ## init logger
    result_filepath = "./output/{}/{}/{}/{}/".format(args.task, args.model, args.dataset, args.seed)
    logger = init_logger(result_filepath)
    logger.info('==' * 27)
    logger.info(args)
    logger.info('==' * 27)
    ## fixed seed
    set_random(args.seed)
    logger.info(f'Using: {args.device}')
    logger.info("Using seed {}.".format(args.seed))
    logger.info('==' * 27)
    ## load data
    if args.task == 'nc':
        args.datapath = "../data/node_classification/{}/".format(args.dataset)
    else:
        raise Exception('This file only do node classifying')
    data = load_data(args)
    args.n_classes = int(data['labels'].max() + 1)
    args.nout = args.n_classes
    if data['features'] is not None and args.use_feats:
        args.nfeat = data['features'].shape[1]
    args.num_nodes = data['features'].shape[0]
    num_edges = len(data['edge_index'][0])
    degrees = get_degrees(data['edge_index'], args.num_nodes).to(args.device)
    args.max_node_degree = int(torch.max(degrees))
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Num classes: {args.n_classes}')
    logger.info(f'Num nodes: {args.num_nodes}')
    logger.info(f'Num edges: {num_edges}')
    logger.info(f'Max degree: {args.max_node_degree}')
    logger.info(f'Feature dim: {args.nfeat}')
    logger.info('==' * 27)
    ## load model and decoder
    if args.use_feats:                                  ## use the initial features
        features = data['features'].to(args.device)
    else:
        args.using_pretrained_feat = True
    model = load_model(args, logger).to(args.device)
    decoder = load_decoder(args, logger)
    logger.info(str(model))
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info(f"Total number of parameters: {tot_params}")


    ## load learning rate, optimizer
    optimizer, scheduler = load_optimizer(args, model)
    logger.info(f'Use Riemannian Adam: {args.using_riemannianAdam}')

    ## train
    t_total = time.time()
    counter = 0
    best_val_metrics = decoder.init_metric_dict()
    best_test_metrics = decoder.init_metric_dict()
    best_emb = None
    edge_index = data['edge_index'].to(args.device)
    grad_file = result_filepath + "grad.txt"
    open(grad_file, 'w').close()
    losses= []
    test_f1s = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        ## get model embeddings
        if args.use_feats:                                  ## use the initial features
            if args.model == 'QGT':
                embeddings = model.forward(edge_index, degrees, features)
            elif args.model == 'NodeFormer':
                embeddings, link_loss_ = model.forward(edge_index, features)
            else:
                embeddings = model.forward(edge_index, features)
        else:                                               ## use random trainable features
            if args.model == 'QGT':
                embeddings = model.forward(edge_index, degrees)
            elif args.model == 'NodeFormer':
                embeddings, link_loss_ = model.forward(edge_index)
            else:
                embeddings = model.forward(edge_index)

        ## decode embeddings to predictions, and calculate loss
        train_idx = data['idx_train']
        output = decoder.decode(embeddings, train_idx)
        labels = data['labels'].to(args.device)
        train_metrics = decoder.compute_metric_loss(output, labels, train_idx)

        ## update parameters
        #with torch.autograd.detect_anomaly():
        if args.model == 'NodeFormer':
            train_metrics['loss'] -= args.lamda * sum(link_loss_) / len(link_loss_)
        losses.append(train_metrics['loss'].item())
        train_metrics['loss'].backward()

        # if args.model == 'QGT':
        #     with open(grad_file, "a") as f:
        #         f.write(f"Epoch: {epoch}\n")
        #         for name, param in model.named_parameters():
        #             if param.grad is not None:
        #                 norm = param.grad.data.norm(2).item()
        #                 f.write(f"{name}: {norm:.6f}\n")
        for param in model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step()
        scheduler.step()

        ## log training results
        t_epoch = time.time() - t
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        if (epoch + 1) % args.log_freq == 0:
            logger.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(t_epoch),
                                   'GPU: {:.1f}MiB'.format(gpu_mem_alloc)
                                   ]))
        
        ## eval (after updating parameters)
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                if args.use_feats:                                  ## use the initial features
                    if args.model == 'QGT':
                        embeddings = model.forward(edge_index, degrees, features)
                    elif args.model == 'NodeFormer':
                        embeddings, link_loss_ = model.forward(edge_index, features)
                    else:
                        embeddings = model.forward(edge_index, features)
                else:                                               ## use random trainable features
                    if args.model == 'QGT':
                        embeddings = model.forward(edge_index, degrees)
                    elif args.model == 'NodeFormer':
                        embeddings, link_loss_ = model.forward(edge_index)
                    else:
                        embeddings = model.forward(edge_index)
                val_idx = data['idx_val']
                output = decoder.decode(embeddings, val_idx)
                val_metrics = decoder.compute_metric_loss(output, labels, val_idx)
                logger.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))

                if decoder.has_improved(best_val_metrics, val_metrics):
                    counter = 0
                    best_val_metrics = val_metrics
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epoch:
                        logger.info("Early stopping")
                        break

        ## test 
        if (epoch + 1) % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                test_idx = data['idx_test']
                output = decoder.decode(embeddings, test_idx)
                test_metrics = decoder.compute_metric_loss(output, labels, test_idx)
                logger.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(test_metrics, 'test')]))
                test_f1s.append(test_metrics['f1'])
                if decoder.has_improved(best_test_metrics, test_metrics):
                    best_test_metrics = test_metrics
                    best_emb = embeddings
        logger.info('==' * 27)
    ## after training
    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    logger.info(" ".join(["Best val results:", format_metrics(best_val_metrics, 'val')]))
    logger.info(" ".join(["Best test results:", format_metrics(best_test_metrics, 'test')]))

    ## save embeddings / model checkpoint
    if args.emb_save:
        output_path = "./output/{}/{}/{}/{}/".format(args.task, args.model, args.dataset, args.seed)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        emb_path = output_path + 'embeddings.npy'
        np.save(emb_path, best_emb.cpu().detach().numpy())
        #logger.info(f"Saved embeddings in {emb_path}")

    if args.model_save:
        output_path = "./output/{}/{}/{}/{}/".format(args.task, args.model, args.dataset, args.seed)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        model_path = output_path + 'model.pth'
        torch.save(model.state_dict(), model_path)
        #logger.info(f"Saved model in {model_path}")

    # save loss and f1 plot
    output_path = "./output/{}/{}/{}/{}/".format(args.task, args.model, args.dataset, args.seed)
    plt.figure(figsize=(10, 5))  # Wider plot
    plt.plot(losses, marker='o', markersize=4, linewidth=2, label='Loss')
    plt.plot(test_f1s, marker='s', markersize=4, linewidth=2, label='f1')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Plot the loss/f1")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(output_path + "/loss_f1_plot.pdf")


if __name__ == '__main__':
    from script.config import args
    from script.utils.train_utils import set_random, init_logger, load_optimizer, format_metrics, lr_schedule
    from script.utils.data_utils import load_data
    from script.utils.preprocessing import get_degrees
    from script.models.load_model import load_model
    from script.models.load_decoder import load_decoder

    warnings.filterwarnings("ignore")
    run(args)
