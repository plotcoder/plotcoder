import random

import numpy as np
import torch

import arguments
import models.data_utils.data_utils as data_utils
import models.model_utils as model_utils
from models.model import PlotCodeGenerator
import filelock
import os
import collections
import time


class Profiler:
	def __init__(self, do_profiling):
		self.time_details = collections.defaultdict(float)
		self.pre_time = time.perf_counter()
		self.do_profiling = do_profiling

	def add_timepoint(self, name):
		if self.do_profiling:
			torch.cuda.current_stream().synchronize()
			curr_time = time.perf_counter()
			if name is not None:
				self.time_details[name] += (curr_time - self.pre_time)
			self.pre_time = curr_time

	def print(self):
		for k, v in self.time_details.items():
			print(k, v)

	def clear(self):
		self.time_details = collections.defaultdict(float)
		self.pre_time = time.perf_counter()


def create_model(args, word_vocab_size, code_vocab):
	model = PlotCodeGenerator(args, word_vocab_size, code_vocab)
	if model.cuda_flag:
		model = model.cuda()
	model_supervisor = model_utils.Supervisor(model, args)
	if args.load_model:
		model_supervisor.load_pretrained(args.load_model)
	else:
		print('Created model with fresh parameters.')
		model_supervisor.model.init_weights(args.param_init)
	return model_supervisor


def file_cache(op, args, path):
	lock_path = os.path.join(args.data_cache_dir, path + ".lock")
	file_path = os.path.join(args.data_cache_dir, path + ".pt")
	with filelock.FileLock(lock_path):
		if not os.path.exists(file_path):
			torch.save(op(), file_path)
		result = torch.load(file_path)
	return result


def train(args):
	print('Training:')
	profiler = Profiler(args.do_profiling)

	data_processor = data_utils.DataProcessor(args)
	profiler.add_timepoint("Initialize DataProcessor")

	train_data = file_cache(lambda: data_processor.load_data(args.train_dataset), args, "train_data")
	train_data, train_indices = file_cache(lambda: data_processor.preprocess(train_data), args, "train_data_processed")
	dev_data = file_cache(lambda: data_processor.load_data(args.dev_dataset), args, "dev_data")
	dev_data, dev_indices = file_cache(lambda: data_processor.preprocess(dev_data), args, "dev_data_processed")
	train_data_size = len(train_data)
	profiler.add_timepoint("Process/Load Data")

	args.code_vocab_size = data_processor.code_vocab_size
	model_supervisor = create_model(args, data_processor.word_vocab_size, data_processor.code_vocab)
	profiler.add_timepoint("Create Model")

	profiler.print()
	profiler.clear()

	logger = model_utils.Logger(args)

	for epoch in range(args.num_epochs):
		random.shuffle(train_data)
		for batch_idx in range(0, train_data_size, args.batch_size):
			# print(epoch, batch_idx)
			batch_input, batch_labels = data_processor.get_batch(train_data, args.batch_size, batch_idx)
			train_loss, train_acc = model_supervisor.train(batch_input, batch_labels)
			# print('train loss: %.4f train acc: %.4f' % (train_loss, train_acc))

			if model_supervisor.global_step % args.eval_every_n == 0:
				model_supervisor.save_model()
				eval_loss, eval_label_acc, eval_data_acc, eval_acc, pred_labels = model_supervisor.eval(dev_data, args.data_order_invariant, args.max_eval_size)
				val_summary = {'train_loss': train_loss, 'train_acc': train_acc, 'eval_loss': eval_loss,
				'eval_label_acc': eval_label_acc, 'eval_data_acc': eval_data_acc, 'eval_acc': eval_acc}
				val_summary['global_step'] = model_supervisor.global_step
				logger.write_summary(val_summary)


def evaluate(args):
	print('Evaluation')
	data_processor = data_utils.DataProcessor(args)
	init_test_data = file_cache(lambda: data_processor.load_data(args.test_dataset), args, "test_data")
	test_data, test_indices = file_cache(lambda: data_processor.preprocess(init_test_data), args, "test_data_processed")

	args.code_vocab_size = data_processor.code_vocab_size
	model_supervisor = create_model(args, data_processor.word_vocab_size, data_processor.code_vocab)
	test_loss, test_label_acc, test_data_acc, test_acc, predictions = model_supervisor.eval(test_data, args.data_order_invariant)

	label_acc_per_category = [0] * args.num_plot_types
	data_acc_per_category = [0] * args.num_plot_types
	acc_per_category = [0] * args.num_plot_types
	cnt_per_category = [0] * args.num_plot_types

	cnt_unpredictable = 0
	for i, item in enumerate(test_data):
		gt_label = item['label']
		if args.joint_plot_types:
			gt_label = data_processor.get_joint_plot_type(gt_label)
		cnt_per_category[gt_label] += 1
		if args.bow:
			pred_label = predictions[i]
			if args.joint_plot_types:
				pred_label = data_processor.get_joint_plot_type(pred_label)
			if pred_label == gt_label:
				label_acc_per_category[gt_label] += 1
			'''
			else:
				print('item index: ', i)
				for key in init_test_data[i]:
					if key == 'context':
						continue
					print('key: ', key)
					print(init_test_data[i][key])
				print('prediction: ', predictions[i])
			'''
		else:
			gt_prog = data_processor.ids_to_prog(item, item['output_gt'])
			pred_prog  = data_processor.ids_to_prog(item, predictions[i])

			pred_label = data_processor.label_extraction(pred_prog)
			if args.joint_plot_types:
				pred_label = data_processor.get_joint_plot_type(pred_label)
			if pred_label == gt_label:
				label_acc_per_category[gt_label] += 1

			target_dfs, target_strs, target_vars = item['target_dfs'], item['target_strs'], item['target_vars']
			pred_dfs, pred_strs, pred_vars, _ = data_processor.data_extraction(pred_prog,
				item['reserved_dfs'], item['reserved_strs'], item['reserved_vars'])
			if args.data_order_invariant:
				if set(target_dfs + target_strs + target_vars) == set(pred_dfs + pred_strs + pred_vars) and \
						len(target_dfs + target_strs + target_vars) == len(pred_dfs + pred_strs + pred_vars):
					cur_data_acc = 1
				else:
					cur_data_acc = 0
			else:
				if target_dfs + target_strs + target_vars == pred_dfs + pred_strs + pred_vars:
					cur_data_acc = 1
				else:
					cur_data_acc = 0
			if cur_data_acc == 1:
				data_acc_per_category[gt_label] += 1
				if pred_label == gt_label:
					acc_per_category[gt_label] += 1

	print('test loss: %.4f test label acc: %.4f test data acc: %.4f test acc: %.4f ' % (test_loss, test_label_acc, test_data_acc, test_acc))
	print('Unpredictable samples: %d' % cnt_unpredictable)
	for i in range(args.num_plot_types):
		print('cnt per category: ', i, cnt_per_category[i])
		if cnt_per_category[i] == 0:
			continue
		print('label acc per category: ', i, label_acc_per_category[i], label_acc_per_category[i] * 1.0 / cnt_per_category[i])
		print('data acc per category: ', i, data_acc_per_category[i], data_acc_per_category[i] * 1.0 / cnt_per_category[i])
		print('acc per category: ', i, acc_per_category[i], acc_per_category[i] * 1.0 / cnt_per_category[i])


if __name__ == "__main__":
	arg_parser = arguments.get_arg_parser('juice')
	args = arg_parser.parse_args()
	args.cuda = not args.cpu and torch.cuda.is_available()
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	if args.eval:
		evaluate(args)
	else:
		train(args)	
