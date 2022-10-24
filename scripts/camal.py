from dataclassses import dataclass
import statistics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import itertools

@dataclass
class DataPoint:
	val: float
	sd: float
	mean: float
	#window: int

class Series(object):
	"""
	Data Series

	Iterable numeric values to be processed to DataPoint for further processing
	"""
	def __init__(self, x, raw_data, window):
		self.x = x
		self.raw_data = raw_data
		self.data = self.extract(window)

	def get_generator (self, window):
		self.data = self.extract(window)

	def get_as_list (self):
		return list(self.data)

	def get_val (self):
		yield point.val for point in self.data

	def get_sd (self):
		yield point.sd for point in self.data

	def get_mean (self):
		yield point.mean for point in self.data

	def get_point_index (self, index):
		return (self.x[index], next(itertools.islice(self.data, index, None)))

	def plot(self, d_type):
		f, ax = plt.subplots()

		if d_type == 0:
			y = list(self.get_val())

		elif d_type == 1:
			y = list(self.get_mean())

		else:
			y = list(self.get_sd())

		ax.plot(self.x, y)
		return ax

	def extract (self, window):
		for ind in range(len(self.raw_data)):
			if ind > window // 2 and ind < len(self.raw_data) - window //2:
				yield DataPoint(
					val = self.raw_data [ind], 
					sd = statistics.stdev(self.raw_data[ind - window // 2 : ind + window // 2]), 
					mean = statistics.mean(self.raw_data[ind - window // 2 : ind + window // 2])
					)

			else:
				yield DataPoint(
					val = self.raw_data[ind],
					sd = None,
					mean = None
					)

	def compare_points (self, ind_1, ind_2, func):
		return func(self.get_point_index(ind_1), self.get_point_index(ind_2))

	def compare_to_past (self, gap, start_pres = 0, end_pres = len(self.x)-1, func):
		if start_pres - gap < 0:
			print("ERROR:: past index < 0")
			return

		if end_pres > len(self.x)-1:
			print("ERROR:: end point is beyond available data")
			return
		
		past = start_pres - gap
		pres = start_pres

		while pres <= end_pres:
			yield compare_points(past, pres, func)

	def converge (self, left, right, func):
		while left != right:
			left, right = self.compare_points(left, right, func)

		return left



	def fit_linear (self, start, end, d_type):
		if d_type == 0:
			y = list(self.get_val())

		elif d_type == 1:
			y = list(self.get_mean())

		else:
			y = list(self.get_sd())


		reg = LinearRegression().fit(self.x, y)

		return reg.score(self.x, y), reg.coef_, reg.intercept_, -reg.intercept_/reg.coef_





