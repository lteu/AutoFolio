import numpy as np
import pandas as pd
from autofolio.facade.af_csv_facade import AFCsvFacade
import yaml
import sys
import os
import json

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"



def run(scenario):
	# perf_fn = "perf.csv"
	# feat_fn = "feats.csv"

	perf_fn = "csv/train/"+scenario+"/runtime.csv"
	feat_fn = "csv/train/"+scenario+"/features.csv"
	description = "csv/train/"+scenario+"/description.txt"

	with open(description) as stream:
		try:
			dic_description = yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	timeout = int(dic_description['algorithm_cutoff_time'])

	# will be created (or overwritten) by AutoFolio
	model_fn = "trash/"+scenario+"_af_model.pkl"
	# print(timeout)

	af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn, maximize=False, objective='runtime',runtime_cutoff=timeout)

	# # fit AutoFolio; will use default hyperparameters of AutoFolio
	af.fit()

	# # tune AutoFolio's hyperparameter configuration for 4 seconds
	config = af.tune(wallclock_limit=900)

	# # evaluate configuration using a 10-fold cross validation
	score = af.cross_validation(config=config)

	# # re-fit AutoFolio using the (hopefully) better configuration
	# # and save model to disk
	af.fit(config=config, save_fn=model_fn)

	# # load AutoFolio model and
	# # get predictions for new meta-feature vector
	# pred = AFCsvFacade.load_and_predict(vec=np.array([1.]), load_fn=model_fn)

	pref_fn = "csv/test/"+scenario+"/features.csv"

	df = pd.read_csv(pref_fn, index_col = 0,delimiter=',')
	# # print(df)
	results ={}
	for i,x in df.iterrows():
		vec = np.array(list(x.values))
		pred = AFCsvFacade.load_and_predict(vec=vec, load_fn=model_fn)
		pred[-1] = pred[-1][0]
		results[i] = pred
		# print(pred)
		# print(i,pred)


	# result_dir = "results-autofolio/"+scenario+".json"
	result_dir = "results-autofolio"
	result_path = result_dir+"/"+scenario+".json"
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	with open(result_path, 'w') as outfile:
		json.dump(results, outfile)


	# print(pred)


def main(args):
	# print(1)
	scenarios =[
		"Caren",
		"Mira",
		"Magnus",
		"Monty",
		"Quill",
		"Bado",
		"Svea",
		"Sora"
	]

	for scenario_name in scenarios:
		# for indx in range(1,11):
		# for indx in range(1,2):
		scenario = scenario_name
		run(scenario)


	

if __name__ == '__main__':
  main(sys.argv[1:])

