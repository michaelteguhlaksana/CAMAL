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
		self.region = {}

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

	def add_region (self, name, index):
		if name in self.region.keys():
			self.region[name].append(index)
		else:
			self.region[name] = [index]


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


class CascadeSeries(object):
	"""
	Several series bundled together to allow different windows and regions all at once
	Assuming all series came from the same set of x
	All series are assumed to be contiguous. Any separation in the series should be separated to 2 series in the args, with the same name
	"""
	def __init__(self, x, series):
		self.x = x
		self.series = {}
		self.hash_index = []
		self.val_index = {}

		for name, ser in series.items():
			self.add_series(name, ser)


	@staticmethod
	def searchInsert(nums: List[int], target: int):
		if len(nums) == 0:
			return 0

		left, right = 0, len(nums) - 1
		
		while left < right:
			mid = (left + right) // 2
			if nums[mid] < target:
				left = mid + 1
			elif nums[mid] > target:
				right = mid - 1
			else:
				return mid

		if nums[left] >= target:
			return left
		else:
			return left + 1

	def add_hash_index (self, name, series):
		mini = min(series.x)
		maxi = max(series.x)

		ind_mini = self.searchInsert(self.hash_index, mini)
		ind_maxi = self.searchInsert(self.hash_index, maxi)

		to_add = [(ind_mini, mini), (ind_maxi, maxi)]

		for ind, add in to_add:
			#shouldn't this be a set then?
			if self.hash_index[ind] != add:
				self.hash_index.insert(ind, add)
		
			if add in self.val_index:
				self.val_index.append(name)

			else:
				self.val_index[add] = [name]
				if 0 < ind < len(self.hash_index)-1:
					prev = self.hash_index[ind - 1]
					aft = self.hash_index[ind + 1]

					#get intersection between the previous and next point to ensure the point contain all series
					self.val_index[add].extend(list( set(self.val_index[prev]) & set(self.val_index[aft]) ))


		for index in range(ind_mini+1, ind_maxi):
			self.val_index[self.hash_index[index]].append(name)

	def add_series (self, name, series):
		if name in self.series.keys():
				self.series [name].append((series, min(series.x), max(series.x)))

			else:
				self.series [name] = [(series, min(series.x), max(series.x))]

			self.add_hash_index(name, series)

	def get_point_series (self, ind, name):
		ser, mini, maxi = self.series[name]

		if mini <= ind <= maxi:
			return ser.get_point_index(ind)

		else:
			return None

	def get_point_all (self, ind):
		nearest_larger = self.searchInsert(self.hash_index, ind)
		if nearest_larger == ind:
			names = self.val_index[ self.hash_index[ind] ]
		else:

			names = list( set(self.val_index[self.hash_index[nearest_larger - 1]]) & set(self.val_index[self.hash_index[nearest_larger]]))

		res = {}

		for name in names:
			res[name] = self.get_point_series(ind, name)

		return res
		





		


