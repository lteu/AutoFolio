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



def run(scenario,path):

	perf_fn = path+"/train/"+scenario+"/runtime.csv"
	feat_fn = path+"/train/"+scenario+"/features.csv"
	description = path+"/train/"+scenario+"/description.txt"

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
	# af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn,runtime_cutoff=timeout)

	# # fit AutoFolio; will use default hyperparameters of AutoFolio
	af.fit(save_fn=model_fn)

	# ===================== TUNING ============================================
	# tune AutoFolio's hyperparameter configuration for 4 seconds
	# config = af.tune(wallclock_limit=60)
	# # evaluate configuration using a 10-fold cross validation
	# score = af.cross_validation(config=config)
	# print('CV score',score)
	# # re-fit AutoFolio using the (hopefully) better configuration
	# # and save model to disk
	# af.fit(config=config, save_fn=model_fn)
	# score = af.cross_validation(config=config)
	# print('CV score',score)
	# =================================================================

	pref_fn = path+"/test/"+scenario+"/features.csv"

	df = pd.read_csv(pref_fn, index_col = 0,delimiter=',')
	# # print(df)
	results ={}
	for i,x in df.iterrows():
		vec = np.array(list(x.values))
		# get predictions for new meta-feature vector
		pred = AFCsvFacade.load_and_predict(vec=vec, load_fn=model_fn)
		pred[-1] = pred[-1][0]
		results[i] = pred

	# result_dir = "results-autofolio/"+scenario+".json"
	# result_dir = "results-autofolio"
	# result_dir = "results-oasc-csv"
	result_dir = "results-oasc-csv-fs"
	result_path = result_dir+"/"+scenario+".json"
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	with open(result_path, 'w') as outfile:
		json.dump(results, outfile)



def main(args):
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

	scenarios +=[
	'MAXSAT19-UCMS',
	'SAT18-EXP',
	'GLUHACK-2018'
	]


	#      scenarios = ["Monty5"]
	# scenario_name = "Caren2"
	# scenario_name = "Quill1"
	path = "csv"
	scenario_name = "Monty5"
	run(scenario_name,path)
	# run(scenario_name)
	# path = "csv"
	# path = "oasc_csv"
	# path = "oasc_csv_fs"

	#       path = "csv"
	#	for scenario_name in scenarios:
	#		run(scenario_name,path)




	

if __name__ == '__main__':
  main(sys.argv[1:])

