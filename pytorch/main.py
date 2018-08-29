import re
import argparse
import os
import shutil
import time
import math
import logging
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher.weights_optimizer import EMAWeightsOptimizer, SWAWeightsOptimizer, MeanWeightsOptimizer


LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


class ArchTrainer(object):
    def __init__(self, context, arch, args, num_classes):
        self.arch = arch
        self.swa = args.swa
        self.num_classes = num_classes
        self.training_log = context.create_train_log("{}_training".format(self.arch))
        self.validation_log = context.create_train_log("{}_validation".format(self.arch))
        self.ema_validation_log = context.create_train_log("{}_ema_validation".format(self.arch))
        self.model = self._create_model()
        self.ema_model = self._create_model(no_grad=True)
        self.ema_weights_optimizer = EMAWeightsOptimizer(self.ema_model, args.ema_decay)

        if self.swa:
            self.swa_validation_log = context.create_train_log("{}_swa_validation".format(self.arch))
            self.mean_validation_log = context.create_train_log("{}_mean_validation".format(self.arch))
            self.ss_swa_validation_log = context.create_train_log("{}_ss_swa_validation".format(self.arch))
            self.ss_mean_validation_log = context.create_train_log("{}_ss_mean_validation".format(self.arch))
            self.swa_model = self._create_model(no_grad=True)
            self.swa_weights_optimizer = SWAWeightsOptimizer(self.swa_model)
            self.mean_model = self._create_model(no_grad=True)
            self.mean_weights_optimizer = MeanWeightsOptimizer(self.mean_model)
            self.ss_swa_model = self._create_model(no_grad=True)
            self.ss_swa_weights_optimizer = SWAWeightsOptimizer(self.ss_swa_model)
            self.ss_mean_model = self._create_model(no_grad=True)
            self.ss_mean_weights_optimizer = MeanWeightsOptimizer(self.ss_mean_model)

        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

    def load(self, checkpoint):
        self.model.load_state_dict(checkpoint['{}_state_dict'.format(self.arch)])
        self.ema_model.load_state_dict(checkpoint['{}_ema_state_dict'.format(self.arch)])
        if self.swa:
            self.swa_model.load_state_dict(checkpoint['{}_swa_state_dict'.format(self.arch)])
            self.swa_weights_optimizer.setNModels(checkpoint['{}_swa_n_models'.format(self.arch)])
            self.mean_model.load_state_dict(checkpoint['{}_mean_state_dict'.format(self.arch)])
            self.ss_swa_model.load_state_dict(checkpoint['{}_swa_state_dict'.format(self.arch)])
            self.ss_swa_weights_optimizer.setNModels(checkpoint['{}_swa_n_models'.format(self.arch)])
            self.ss_mean_model.load_state_dict(checkpoint['{}_ss_mean_state_dict'.format(self.arch)])
        self.optimizer.load_state_dict(checkpoint['{}_optimizer'.format(self.arch)])

    def train(self, train_loader, train_len, is_swa_epoch, is_ss_swa_epoch, epoch):
        LOG.info("Training architecture {}".format(self.arch))
        if is_swa_epoch and self.swa_weights_optimizer.getNModels() == 0:
            self.swa_weights_optimizer.update(self.model)
            self.ss_swa_weights_optimizer.update(self.model)
        train(train_loader, train_len, self.model, self.ema_model, self.swa_model, self.ss_swa_model,
              is_swa_epoch, is_ss_swa_epoch, self.optimizer, self.ema_weights_optimizer, self.swa_weights_optimizer,
              self.ss_swa_weights_optimizer, epoch, self.training_log)
        if is_swa_epoch:
            LOG.info("Updating the MEAN model")
            self.mean_weights_optimizer.update(self.ema_model, self.swa_model)
        if is_ss_swa_epoch:
            LOG.info("Updating the Super-Sampling MEAN model")
            self.ss_mean_weights_optimizer.update(self.ema_model, self.ss_swa_model)

    def evaluate(self, eval_loader, eval_len, is_swa_epoch, is_ss_swa_epoch, global_step, epoch, train_loader=None, train_len=0):
        LOG.info("Evaluating architecture {}".format(self.arch))
        LOG.info("Evaluating the primary model:")
        prec1 = validate(eval_loader, eval_len, self.model, self.validation_log, global_step, epoch)
        LOG.info("Evaluating the EMA model:")
        ema_prec1 = validate(eval_loader, eval_len, self.ema_model, self.ema_validation_log, global_step, epoch)
        if is_swa_epoch:
            if train_loader:
                LOG.info("Updating SWA Batch Norm")
                batch_norm_update(train_loader, train_len, self.swa_model)
                LOG.info("Updating MEAN Batch Norm")
                batch_norm_update(train_loader, train_len, self.mean_model)
            LOG.info("Evaluating the SWA model:")
            swa_prec1 = validate(eval_loader, eval_len, self.swa_model, self.swa_validation_log, global_step, epoch)
            LOG.info("Evaluating the MEAN model:")
            mean_prec1 = validate(eval_loader, eval_len, self.mean_model, self.mean_validation_log, global_step, epoch)
        else:
            swa_prec1 = 0
            mean_prec1 = 0
        if is_ss_swa_epoch:
            if train_loader:
                LOG.info("Updating Super-Sampling SWA Batch Norm")
                batch_norm_update(train_loader, train_len, self.ss_swa_model)
                LOG.info("Updating Super-Sampling MEAN Batch Norm")
                batch_norm_update(train_loader, train_len, self.ss_mean_model)
            LOG.info("Evaluating the Super-Sampling SWA model:")
            ss_swa_prec1 = validate(eval_loader, eval_len, self.ss_swa_model, self.ss_swa_validation_log, global_step, epoch)
            LOG.info("Evaluating the Super-Sampling MEAN model:")
            ss_mean_prec1 = validate(eval_loader, eval_len, self.ss_mean_model, self.ss_mean_validation_log, global_step, epoch)
        else:
            ss_swa_prec1 = 0
            ss_mean_prec1 = 0
        return max(ema_prec1, swa_prec1, ss_swa_prec1, mean_prec1, ss_mean_prec1)

    def get_state(self):
        return {
            '{}_state_dict'.format(self.arch): self.model.state_dict(),
            '{}_ema_state_dict'.format(self.arch): self.ema_model.state_dict(),
            '{}_swa_state_dict'.format(self.arch): self.swa_model.state_dict() if self.swa else self.ema_model.state_dict(),
            '{}_swa_n_models'.format(self.arch): self.swa_weights_optimizer.getNModels() if self.swa else 0,
            '{}_mean_state_dict'.format(self.arch): self.mean_model.state_dict() if self.swa else self.ema_model.state_dict(),
            '{}_ss_swa_state_dict'.format(self.arch): self.ss_swa_model.state_dict() if self.swa else self.ema_model.state_dict(),
            '{}_ss_swa_n_models'.format(self.arch): self.ss_swa_weights_optimizer.getNModels() if self.swa else 0,
            '{}_ss_mean_state_dict'.format(self.arch): self.ss_mean_model.state_dict() if self.swa else self.ema_model.state_dict(),
            '{}_optimizer'.format(self.arch): self.optimizer.state_dict()
        }

    def _create_model(self, no_grad=False, pretrained=False):
        LOG.info("=> creating {no_grad}model '{arch}'".format(
            no_grad='(no_grad) ' if no_grad else '',
            arch=self.arch))

        model_factory = architectures.__dict__[self.arch]
        model_params = dict(pretrained=pretrained, num_classes=self.num_classes)
        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()

        if no_grad: #Save memory if no need to save gradient
            for param in model.parameters():
                param.detach_()

        return model


def main(context):
    global global_step
    global best_prec1

    checkpoint_path = context.transient_dir

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    ss_swa_cycles = 1 if args.swa_cycle < 5 else 2 if args.swa_cycle < 10 else (args.swa_cycle / 5)

    arch_trainers = [ArchTrainer(context, arch, args, num_classes) for arch in args.arch]

    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)
    train_len = len(train_loader)
    eval_len = len(eval_loader)

    for arch_trainer in arch_trainers:
        LOG.info(parameters_string(arch_trainer.model))

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        for arch_trainer in arch_trainers:
            arch_trainer.load(checkpoint)
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    LOG.info("Setting seed to {}".format(args.seed))
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.evaluate:
        for arch_trainer in arch_trainers:
            arch_trainer.evaluate(eval_loader, eval_len, True, True, global_step, args.start_epoch)
        LOG.info("Evaluating all architectures")
        validate_all(eval_loader, eval_len, arch_trainers, True, True, global_step, args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        is_swa_epoch = args.swa and epoch >= args.swa_start and\
                       ((epoch - args.swa_start) % args.swa_cycle == 0 or (epoch + 1) == args.epochs)
        is_ss_swa_epoch = args.swa and epoch >= args.swa_start and\
                       ((epoch - args.swa_start) % ss_swa_cycles == 0 or (epoch + 1) == args.epochs)
        is_ss_swa_epoch = is_ss_swa_epoch or is_swa_epoch

        for arch_trainer in arch_trainers:
            arch_trainer.train(train_loader, train_len, is_swa_epoch, is_ss_swa_epoch, epoch)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and ((epoch + 1) % args.evaluation_epochs == 0 or is_ss_swa_epoch):
            start_time = time.time()
            curr_bests = [arch_trainer.evaluate(eval_loader, eval_len, is_swa_epoch, is_ss_swa_epoch, global_step,
                                                epoch + 1, train_loader, train_len) for arch_trainer in arch_trainers]
            if is_ss_swa_epoch:
                LOG.info("Evaluating all architectures")
                all_prec = validate_all(eval_loader, eval_len, arch_trainers, is_swa_epoch, is_ss_swa_epoch, global_step, args.start_epoch)
            else:
                all_prec = 0
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            curr_best = max(curr_bests)
            is_best = curr_best > best_prec1 or all_prec > best_prec1
            if is_best:
                best_prec1 = max(curr_best,all_prec)
                LOG.info("--- This is the best score yet ({})---".format(best_prec1))
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            curr_state = { 'epoch': epoch + 1, 'global_step': global_step, 'best_prec1': best_prec1, 'arch': ','.join(args.arch) }
            for arch_trainer in arch_trainers:
                curr_state.update(arch_trainer.get_state())
            save_checkpoint(curr_state, is_best, checkpoint_path, epoch + 1)
    LOG.info(datetime.datetime.now().strftime('Finished main at %Y%m%d_%H%M%S'))


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader


def train(train_loader, train_len, model, ema_model, swa_model, ss_swa_model, is_swa_epoch, is_ss_swa_epoch,
          optimizer, ema_weights_optimizer, swa_weights_optimizer, ss_swa_weights_optimizer, epoch, log):
    global global_step

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()
    if is_swa_epoch:
        swa_model.train()
    if is_ss_swa_epoch:
        ss_swa_model.train()

    end = time.time()
    for i, ((input, ema_input, swa_input, ss_swa_input), target) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        lr = adjust_learning_rate(epoch, i, train_len)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        meters.update('lr', lr)

        input_var = torch.autograd.Variable(input)
        ema_input_var = torch.autograd.Variable(ema_input, volatile=True)
        swa_input_var = torch.autograd.Variable(swa_input, volatile=True)
        ss_swa_input_var = torch.autograd.Variable(ss_swa_input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        if is_ss_swa_epoch:
            ss_swa_model_out = ss_swa_model(ss_swa_input_var)
        if is_swa_epoch:
            swa_model_out = swa_model(swa_input_var)
        ema_model_out = ema_model(ema_input_var)
        model_out = model(input_var)

        if isinstance(model_out, Variable):
            assert args.logit_distance_cost < 0
            logit1 = model_out
            ema_logit = ema_model_out
            if is_swa_epoch:
                swa_logit = swa_model_out
            if is_ss_swa_epoch:
                ss_swa_logit = ss_swa_model_out
        else:
            assert len(model_out) == 2
            assert len(ema_model_out) == 2
            logit1, logit2 = model_out
            ema_logit, _ = ema_model_out
            if is_swa_epoch:
                assert len(swa_model_out) == 2
                swa_logit, _ = swa_model_out
            if is_ss_swa_epoch:
                assert len(ss_swa_model_out) == 2
                ss_swa_logit, _ = ss_swa_model_out

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
        if is_swa_epoch:
            swa_logit = Variable(swa_logit.detach().data, requires_grad=False)
        if is_ss_swa_epoch:
            ss_swa_logit = Variable(ss_swa_logit.detach().data, requires_grad=False)

        if args.logit_distance_cost >= 0:
            class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.data[0])
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0

        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        meters.update('class_loss', class_loss.data[0])

        ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.data[0])

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            if is_swa_epoch:
                consistency_weight /= 3
            elif is_ss_swa_epoch:
                consistency_weight /= 2
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
            meters.update('cons_loss', consistency_loss.data[0])
            swa_consistency_loss = (consistency_weight * consistency_criterion(cons_logit, swa_logit) / minibatch_size) if is_swa_epoch else 0
            ss_swa_consistency_loss = (consistency_weight * consistency_criterion(cons_logit, ss_swa_logit) / minibatch_size) if is_ss_swa_epoch else 0
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)
            swa_consistency_loss = 0
            ss_swa_consistency_loss = 0

        loss = class_loss + consistency_loss + res_loss + swa_consistency_loss + ss_swa_consistency_loss
        assert not (np.isnan(loss.data[0]) or loss.data[0] > 2e6), 'Loss explosion: {}'.format(loss.data[0])
        meters.update('loss', loss.data[0])

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100. - prec5[0], labeled_minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
        meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)
        meters.update('ema_top5', ema_prec5[0], labeled_minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        ema_weights_optimizer.update(model, global_step)
        if is_swa_epoch:
            swa_weights_optimizer.update(model)
        if is_ss_swa_epoch:
            ss_swa_weights_optimizer.update(model)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, train_len, meters=meters))
            log.record(epoch + i / train_len, {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })


def validate(eval_loader, eval_len, model, log, global_step, epoch):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        output1, output2 = model(input_var)
        softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        meters.update('class_loss', class_loss.data[0], labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    i, eval_len, meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })

    return meters['top1'].avg


def validate_all(eval_loader, eval_len, arch_trainers, is_swa_epoch, is_ss_swa_epoch, global_step, epoch):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    assert(is_swa_epoch or is_ss_swa_epoch)
    for arch_trainer in arch_trainers:
        if is_swa_epoch:
            arch_trainer.mean_model.eval()
        if is_ss_swa_epoch:
            arch_trainer.ss_mean_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        outputs1 = []
        outputs2 = []
        for arch_trainer in arch_trainers:
            if is_swa_epoch:
                mean_out1, mean_out2 = arch_trainer.mean_model(input_var)
                outputs1.append(mean_out1)
                outputs2.append(mean_out2)
            if is_ss_swa_epoch:
                ss_mean_out1, ss_mean_out2 = arch_trainer.ss_mean_model(input_var)
                outputs1.append(ss_mean_out1)
                outputs2.append(ss_mean_out2)
        softmaxs1 = [F.softmax(output1, dim=1) for output1 in outputs1]
        softmaxs2 = [F.softmax(output2, dim=1) for output2 in outputs2]
        #output1 = [float(sum(col))/len(col) for col in zip(*outputs1)]
        output1 = torch.stack(outputs1, 0).mean(0)
        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        meters.update('class_loss', class_loss.data[0], labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    i, eval_len, meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))

    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(epoch, step_in_epoch, total_steps_in_epoch):
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    if args.swa and epoch >= args.swa_start:
        lr = args.swa_lr
    else:
        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)
    return lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def batch_norm_update(train_loader, train_len, model):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for i, ((input, _, _, _), _) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b
        if i % args.print_freq == 0:
            LOG.info('Batch Normalizing: [{0}/{1}]'.format(i, train_len))

    model.apply(lambda module: _set_momenta(module, momenta))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOG.info("Running with {} GPUs".format(torch.cuda.device_count()))
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))

