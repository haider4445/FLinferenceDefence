import argparse

def parser_func():

	parser = argparse.ArgumentParser(description = "Defense")
	parser.add_argument('--DataFile', type = str, nargs = "?", default = "news_cleaned.csv", help = 'data file path')
	parser.add_argument('--LogFile', type = str, nargs = "?", default = "news.log", help = 'log file path')
	parser.add_argument('--n_attacker', type = int, nargs = "?", default = 30, help = 'attacker features')
	parser.add_argument('--num_target_features', type = int, nargs = "?", default = 29, help = 'other features')
	parser.add_argument('--num_exps', type = int, nargs = "?", default = 2, help = 'running times')


	parser.add_argument('--train_percentage', type = float, nargs = "?", default = 0.6, help = 'train portion')
	parser.add_argument('--test_percentage', type = float, nargs = "?", default = 0.2, help = 'test portion')
	parser.add_argument('--pred_percentage', type = float, nargs = "?", default = 0.2, help = 'predict portion')
	
	parser.add_argument('--enableConfRound', type = bool, nargs = "?", default = False, help = 'Defense type')
	parser.add_argument('--EnablePREDVEL', type = bool, nargs = "?", default = False, help = 'Defense type')
	parser.add_argument('--EnableNoising', type = bool, nargs = "?", default = False, help = 'Defense type')
	parser.add_argument('--EnableDP', type = bool, nargs = "?", default = False, help = 'Defense Type')
	

	parser.add_argument('--DPEpsilon', type = float, nargs = "?", default = 1, help = 'Epsilon DP')
	parser.add_argument('--DPDelta', type = float, nargs = "?", default = 1e-5, help = 'Delta DP')
	parser.add_argument('--DPSensitivity', type = float, nargs = "?", default = 0.01, help = 'Sensitivity DP')
	

	parser.add_argument('--roundPrecision', type = float, nargs = "?", default = 1, help = 'RoundingPrecision')
	parser.add_argument('--StdDevNoising', type = float, nargs = "?", default = 0.1, help = 'StdDevNoising')
	

	parser.add_argument('--epoch_num', type = int, nargs = "?", default = 60, help = 'Epochs')
	parser.add_argument('--epochsForClassifier', type = int, nargs = "?", default = 60, help = 'Epochs')

	parser.add_argument('--meanLambda', type = float, nargs = "?", default = 1.2, help = 'KnownMeanLambda')
	
	parser.add_argument('--enableAttackerFeatures', type = bool, nargs = "?", default = True, help = 'EnableAttackerFeatures')
	parser.add_argument('--enableMean', type = bool, nargs = "?", default = False, help = 'EnableKnownMeanConstraint')
	parser.add_argument('--unknownVarLambda', type = float, nargs = "?", default = 0.25, help = 'UnknownVarLambda')
	
	parser.add_argument('--perturbation_level', type = float, nargs = "?", default = -4, help = 'perturbation_level')	
	parser.add_argument('--outputDim', type = int, nargs = "?", default = 5, help = 'ClassNum')	
	parser.add_argument('--modelType', type = str, nargs = "?", default = "NN", help = 'ModeType')


	parser.add_argument('--max_depth', type = int, nargs = "?", default = 5, help = 'max_depth')	
	parser.add_argument('--min_samples_split', type = int, nargs = "?", default = 3, help = 'min_samples_split')	
	parser.add_argument('--min_samples_leaf', type = int, nargs = "?", default = 5, help = 'min_samples_leaf')		
	parser.add_argument('--min_impurity_decrease', type = float, nargs = "?", default = 1e-5, help = 'node split if induces a decrease of the impurity greater than or equal to this value')	
	parser.add_argument('--criterion', type = str, nargs = "?", default = "gini", help = 'function to measure quality of split, must in {gini, entropy}')	
	

	return parser