import argparse

def parser_func():

	parser = argparse.ArgumentParser(description = "Defense")
	parser.add_argument('--DataFile', type = str, nargs = "?", default = "news_cleaned.csv", help = 'data file path')
	parser.add_argument('--LogFile', type = str, nargs = "?", default = "news.log", help = 'log file path')
	parser.add_argument('--n_attacker', type = int, nargs = "?", default = 30, help = 'attacker features')
	parser.add_argument('--n_victim', type = int, nargs = "?", default = 29, help = 'other features')
	parser.add_argument('--runtimes', type = int, nargs = "?", default = 2, help = 'running times')


	parser.add_argument('--trainpart', type = float, nargs = "?", default = 0.6, help = 'train portion')
	parser.add_argument('--testpart', type = float, nargs = "?", default = 0.2, help = 'test portion')
	parser.add_argument('--predictpart', type = float, nargs = "?", default = 0.2, help = 'predict portion')
	
	parser.add_argument('--enableConfRound', type = bool, nargs = "?", default = False, help = 'Defense type')
	parser.add_argument('--EnablePREDVEL', type = bool, nargs = "?", default = False, help = 'Defense type')
	parser.add_argument('--EnableNoising', type = bool, nargs = "?", default = False, help = 'Defense type')
	parser.add_argument('--EnableDropout', type = bool, nargs = "?", default = False, help = 'Defense Type')
	parser.add_argument('--EnableDP', type = bool, nargs = "?", default = False, help = 'Defense Type')
	parser.add_argument('--EnableEncryption', type = bool, nargs = "?", default = False, help = 'Defense Type')
	
	parser.add_argument('--EnablePREDVELRankingOnly', type = bool, nargs = "?", default = False, help = 'Defense type')

	parser.add_argument('--DPEpsilon', type = float, nargs = "?", default = 1, help = 'Epsilon DP')
	parser.add_argument('--DPDelta', type = float, nargs = "?", default = 1e-5, help = 'Delta DP')
	parser.add_argument('--DPSensitivity', type = float, nargs = "?", default = 0.01, help = 'Sensitivity DP')
	

	parser.add_argument('--roundPrecision', type = float, nargs = "?", default = 1, help = 'RoundingPrecision')
	parser.add_argument('--StdDevNoising', type = float, nargs = "?", default = 0.1, help = 'StdDevNoising')
	

	parser.add_argument('--epochsForAttack', type = int, nargs = "?", default = 60, help = 'Epochs')
	parser.add_argument('--epochsForClassifier', type = int, nargs = "?", default = 60, help = 'Epochs')

	parser.add_argument('--meanLambda', type = float, nargs = "?", default = 1.2, help = 'KnownMeanLambda')
	
	parser.add_argument('--enableAttackerFeatures', type = bool, nargs = "?", default = True, help = 'EnableAttackerFeatures')
	parser.add_argument('--enableMean', type = bool, nargs = "?", default = False, help = 'EnableKnownMeanConstraint')
	parser.add_argument('--unknownVarLambda', type = float, nargs = "?", default = 0.25, help = 'UnknownVarLambda')
	
	parser.add_argument('--perturbation_level', type = float, nargs = "?", default = -4, help = 'perturbation_level')	
	parser.add_argument('--outputDim', type = int, nargs = "?", default = 5, help = 'ClassNum')	
	parser.add_argument('--modelType', type = str, nargs = "?", default = "NN", help = 'ModeType')
	

	parser.add_argument('--results_file_path', type = str, nargs = "?", default = "/content/drive/My Drive/results_output_GRNA.csv", help = 'results output path')

	return parser