import numpy as np
import pandas as pd
from autofolio.facade.af_csv_facade import AFCsvFacade

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"

# perf_fn = "perf.csv"
# feat_fn = "feats.csv"

perf_fn = "examples/caren/train_runtime.csv"
feat_fn = "examples/caren/train_features.csv"


# will be created (or overwritten) by AutoFolio
model_fn = "af_model.pkl"

af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn, maximize=False, objective='runtime',runtime_cutoff=1200)

# fit AutoFolio; will use default hyperparameters of AutoFolio
af.fit()

# tune AutoFolio's hyperparameter configuration for 4 seconds
config = af.tune(wallclock_limit=2)

# evaluate configuration using a 10-fold cross validation
score = af.cross_validation(config=config)

# re-fit AutoFolio using the (hopefully) better configuration
# and save model to disk
af.fit(config=config, save_fn=model_fn)

# load AutoFolio model and
# get predictions for new meta-feature vector
# pred = AFCsvFacade.load_and_predict(vec=np.array([1.]), load_fn=model_fn)

pref_fn = "examples/caren/test_features.csv"

df = pd.read_csv(pref_fn, index_col = 0,delimiter=',')
# print(df)
for i,x in df.iterrows():
	vec = np.array(list(x.values))
	pred = AFCsvFacade.load_and_predict(vec=vec, load_fn=model_fn)
	print(pred)
	# print(i,pred)

# print(pred)


