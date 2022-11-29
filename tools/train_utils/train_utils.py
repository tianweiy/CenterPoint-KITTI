import glob
import os
import numpy

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import time
# from memory_profiler import profile

def val_one_epoch(model, test_loader, model_func, cur_epoch, total_epochs, total_it_each_epoch, val_iter,
                  tbar, tb_log):
    """ Exactly what the name says.

    Args:
        tbar ([type]): [For printing]
        tb_log ([type]): [For tensorboard]
    """
    total = []
    test_dataloader_iter = iter(test_loader)
    pbar = tqdm.tqdm(total=total_it_each_epoch, leave=(cur_epoch + 1 == total_epochs), desc='val', dynamic_ncols=True)
    for it in range(len(test_loader)):
        val_iter += 1
        batch_test = next(test_dataloader_iter)
        model.eval()
        with torch.no_grad():
            res, _ = model_func(model, batch_test)
        model.train()
        preds = [torch.mean(z["pred_scores"]).item() for z in res if z["pred_scores"].nelement()]
        val_loss = 1 - numpy.mean(preds) if len(preds) > 0 else 1
        total.append(val_loss)    
        pbar.set_postfix(dict(total_it=val_iter))
        pbar.update()
        tbar.set_postfix({"val_loss": val_loss})
        tbar.refresh()
        tb_log.add_scalar('val/loss_per_step', val_loss, val_iter) 
    val_loss = numpy.mean(total) if len(total) > 0 else model.val_loss
    tb_log.add_scalar('val/loss_per_epoch', val_loss, cur_epoch) 
    model.val_loss = val_loss
    

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, **kwargs):


    start_time = time.time()
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
   
    
    accumulated_loss = 0
    for cur_it in range(total_it_each_epoch):
       
        epoch_start = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        
        # optimizer.zero_grad() - Optimization 
        for param in model.parameters():
            param.grad = None
       
        loss, tb_dict, disp_dict = model_func(model, batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        
        disp_dict.update({'loss': loss.item(), 'val_loss': model.val_loss, 'lr': cur_lr})
        accumulated_loss += loss.item()
        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
        tb_log.add_scalar('time/one_batch_sec', time.time() - epoch_start, accumulated_iter)
        accumulated_iter += 1
       

    
    if rank == 0:
        pbar.close()
    tb_log.add_scalar('train/loss_per_epoch_2', accumulated_loss / total_it_each_epoch, kwargs["cur_epoch"])
    tb_log.add_scalar('train/loss_per_epoch', accumulated_loss / total_it_each_epoch, accumulated_iter)
    tb_log.add_scalar('time/one_epoch_sec', time.time() - start_time, accumulated_iter)
    return accumulated_iter

def train_model(model, optimizer, train_loader, test_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, **kwargs):
    """
    [Training of the model]
    
    """
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        test_dataloader_iter = iter(test_loader)

        for cur_epoch in tbar:
            
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
                if kwargs.get("test_sampler", None):
                    kwargs["test_sampler"].set_epoch(cur_epoch)
                    
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            val_iter = accumulated_iter
            
            # Train one epoch, and return the accumulated iteration for logging purposes.
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                **kwargs,
                cur_epoch = cur_epoch,
                total_epochs=total_epochs
            )
        
            # Calculate one epoch of the validation loss.
            if (cur_epoch + 1) % 2 == 0 or cur_epoch + 1 == total_epochs:
                val_one_epoch(model, test_loader, model_func, cur_epoch, total_epochs, total_it_each_epoch, val_iter,
                    tbar, tb_log)

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
           
def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
