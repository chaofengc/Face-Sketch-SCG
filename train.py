from utils.timer import Timer
from utils.logger import Logger
from utils import utils

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from evaluate import evaluate

import torch 
import os


def train(opt):
        
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
        
    logger = Logger(opt) # only use logger for the first process
    timer = Timer()

    opt.total_epochs = opt.n_epochs 

    single_epoch_iters = (dataset_size // opt.batch_size)
    total_iters = opt.total_epochs * single_epoch_iters 
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters
    start_iter = opt.resume_iter
    print('Start training from epoch: {:05d}; iter: {:07d}'.format(opt.resume_epoch, opt.resume_iter))
    for epoch in range(opt.resume_epoch, opt.n_epochs):   
        for i, data in enumerate(dataset, start=start_iter):
            cur_iters += 1
            logger.set_current_iter(cur_iters)
            # =================== load data ===============
            model.set_input(data, cur_iters)
            timer.update_time('DataTime')
    
            # =================== model train ===============
            model.forward(), timer.update_time('Forward')
            model.optimize_parameters(), timer.update_time('Backward')
            loss = model.get_current_losses()
            loss.update(model.get_lr())
            logger.record_losses(loss)

            # =================== save model and visualize ===============
            if cur_iters % opt.print_freq == 0:
                print('Model log directory: {}'.format(opt.expr_dir))
                epoch_progress = '{:03d}|{:05d}/{:05d}'.format(epoch, i, single_epoch_iters)
                logger.printIterSummary(epoch_progress, cur_iters, total_iters, timer)
    
            if cur_iters % opt.visual_freq == 0:
                print('Visualizing the results......')
                visual_imgs = model.get_current_visuals()
                logger.record_images(visual_imgs)
                if hasattr(model, 'get_att_maps'):
                    visual_attmaps = model.get_att_maps()
                    if visual_attmaps:
                        logger.record_images(visual_attmaps, tag='attmaps')

            if cur_iters % opt.save_iter_freq == 0:
                print('saving current model (epoch %d, iters %d)' % (epoch, cur_iters))
                save_suffix = 'iter_%d' % cur_iters 
                info = {'resume_epoch': epoch, 'resume_iter': i+1}
                model.save_networks(save_suffix, info)

                sketch_out_scores, photo_out_scores = evaluate(model, opt)
                logger.record_scalar({'smetric/fsim': sketch_out_scores[0], 'smetric/lpips': sketch_out_scores[1], 'smetric/dists': sketch_out_scores[2]})
                logger.record_scalar({'pmetric/fsim': photo_out_scores[0], 'pmetric/lpips': photo_out_scores[1], 'pmetric/dists': photo_out_scores[2]})

            if cur_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, iters %d)' % (epoch, cur_iters))
                info = {'resume_epoch': epoch, 'resume_iter': i+1}
                model.save_networks('latest', info)

            if i >= single_epoch_iters - 1:
                start_iter = 0
                break

            if opt.debug: break
        
        model.update_learning_rate()
    logger.close()


if __name__ == '__main__':
    opt = TrainOptions().parse()
    train(opt)