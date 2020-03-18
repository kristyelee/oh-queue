#!/usr/bin/env python3
import calendar
import csv
import ctypes
import itertools
import math
import operator
import os
import re
import numpy

from collections import namedtuple
from datetime import datetime, timedelta, tzinfo
from pprint import pprint
from timeit import default_timer
from functools import reduce, partial

def call_without_ctrl_c_handler_replacement(func):  # On Python 2.7 on Windows, NumPy messes up Ctrl+C handling. This function fixes it. It won't change anything on other systems.
	try:
		SetConsoleCtrlHandler_body_new = b'\xC2\x08\x00' if ctypes.sizeof(ctypes.c_void_p) == 4 else b'\xC3'
		try: SetConsoleCtrlHandler_body = (lambda kernel32: (lambda pSetConsoleCtrlHandler:
			kernel32.VirtualProtect(pSetConsoleCtrlHandler, ctypes.c_size_t(1), 0x40, ctypes.byref(ctypes.c_uint32(0)))
			and (ctypes.c_char * 3).from_address(pSetConsoleCtrlHandler.value)
		)(ctypes.cast(kernel32.SetConsoleCtrlHandler, ctypes.c_void_p)))(ctypes.windll.kernel32)
		except: SetConsoleCtrlHandler_body = None
		if SetConsoleCtrlHandler_body:
			SetConsoleCtrlHandler_body_old = SetConsoleCtrlHandler_body[0:len(SetConsoleCtrlHandler_body_new)]
			SetConsoleCtrlHandler_body[0:len(SetConsoleCtrlHandler_body_new)] = SetConsoleCtrlHandler_body_new
		try:
			return func()
		finally:
			if SetConsoleCtrlHandler_body:
				SetConsoleCtrlHandler_body[0:len(SetConsoleCtrlHandler_body_new)] = SetConsoleCtrlHandler_body_old
	except ImportError as e:
		pass

def import_numpy(): import numpy, numpy.fft; return numpy
numpy = call_without_ctrl_c_handler_replacement(import_numpy)
numpy_fftpack_lite = getattr(numpy.fft, 'fftpack_lite', None)

def fftpack_lite_rfftb(buf, s, scratch=None):
	n = len(buf)
	m = (n - 1) * 2
	temp = numpy.empty(m, buf.dtype) if scratch is None else scratch if scratch.dtype == buf.dtype else scratch.view(buf.dtype)
	numpy.divide(buf, m, temp[:n])
	temp[n:m] = 0
	result = (numpy_fftpack_lite.rfftb if numpy_fftpack_lite is not None else numpy.fft.irfft)(temp[0:m], s)
	if numpy_fftpack_lite is None:
		result *= s
	return result

def array_lexicographical_compare(a, b, less=numpy.less, scratch=None):
	# 1D arrays only!
	# 'less' parameter is optional
	# Tests:
	#  assert array_lexicographical_compare([0, 0], [1]) < 0
	#  assert array_lexicographical_compare([0], [0, 0]) < 0
	#  assert array_lexicographical_compare([1], [0]) > 0
	#  assert array_lexicographical_compare([0], [0]) == 0
	#  assert array_lexicographical_compare([0], [1]) < 0
	#  assert array_lexicographical_compare([0, 0], [0]) > 0
	#  assert array_lexicographical_compare([1], [0, 0]) > 0
	an = len(a)
	bn = len(b)
	n = bn if bn < an else an
	x = a[:n] if an > n else a
	y = b[:n] if bn > n else b
	if scratch is not None: scratch = scratch[:n]
	scratch = less(x, y, scratch) if less is not None else x < y
	i = scratch.argmax() if n > 0 else 0
	if 0 == i < n and not (a[i] < b[i]): i = n
	scratch = less(y, x, scratch) if less is not None else y < x
	j = scratch.argmax() if n > 0 else 0
	if 0 == j < n and not (b[j] < a[j]): j = n
	if i < j: c = -1
	elif i > j: c = +1
	elif an < bn: c = -1
	elif an > bn: c = +1
	else: c = 0
	return c

def fftpad(v, m, padded):
	vn = len(v)
	if vn < m or padded.dtype != v.dtype:
		padded[vn:m] = 0
		padded[0:vn] = v
		v = padded[:m]
	return v

def fftconvolve(x, y, x_y_transform_cache=None, pad=fftpad, initialize=numpy_fftpack_lite.rffti if numpy_fftpack_lite is not None else {}.get(None), forward=numpy_fftpack_lite.rfftf if numpy_fftpack_lite is not None else numpy.fft.rfft, multiply=numpy.multiply, backward=fftpack_lite_rfftb, cache=[]):
	cn = max(len(x) + len(y) - 1, 0)
	mlog2 = cn.bit_length()
	m = 1 << mlog2
	while len(cache) <= mlog2:
		cache.append(None)
	entry = cache[mlog2]
	if entry is None:
		cache[mlog2] = (s, padded) = (initialize(m) if initialize is not None else m, numpy.empty(m * 2, float))
	else:
		(s, padded) = cache[mlog2]
	do_pad = partial(pad, m=m, padded=padded)
	a = forward(do_pad(x), s) if x_y_transform_cache is None or len(x_y_transform_cache) <= 0 or x_y_transform_cache[0] is None else x_y_transform_cache[0]
	b = forward(do_pad(y), s) if x_y_transform_cache is None or len(x_y_transform_cache) <= 1 or x_y_transform_cache[1] is None else x_y_transform_cache[1]
	if x_y_transform_cache is not None:
		if len(x_y_transform_cache) > 0: x_y_transform_cache[0] = a
		if len(x_y_transform_cache) > 1: x_y_transform_cache[1] = b
	c = backward(multiply(a, b, padded[:m + 2].view(complex)), s, scratch=padded)
	return c[:cn]

def overlap_add_convolve(x, y, convolve, out=None, direct_convolve=numpy.convolve):
	xn = len(x)
	yn = len(y)
	if yn < xn: (x, xn, y, yn) = (y, yn, x, xn)
	use_direct = direct_convolve is not None and xn < 0x20
	if use_direct or 2 * xn >= yn:
		result = (direct_convolve if use_direct else convolve)(x, y)
		if out is not None:
			out[:len(result)] = result
		else:
			out = result
		return out
	zn = xn + yn - 1 if xn > 0 and yn > 0 else 0
	if out is None:
		out = numpy.empty(zn, float)
		assert out.__setitem__(slice(None), numpy.nan) is None or True  # in debug mode, fill with NaNs to find any bugs
	elif len(out) > zn:
		out = out[:zn]
	if xn > 0 and yn > 0:
		saved_transforms = [None]
		blocksize = 1 << min((xn - 1).bit_length(), (yn - 1).bit_length())
		saved = None
		i = 0
		while i < yn:
			j = i + blocksize
			if j > yn:
				j = yn
				saved_transforms[0] = None
			z = convolve(x[0:xn], y[i:j], saved_transforms)
			out[i : j + xn - 1] = z
			if saved is not None:
				out[i : i + len(saved)] += saved
			saved = z[blocksize : blocksize * 2 -1]
			i = j
	return out

def overlap_add_test(ntests):
	import numpy.random
	for _ in range(ntests):
		a = numpy.asarray(numpy.random.randint(0, 8, numpy.random.randint(1, 16)), float)
		b = numpy.asarray(numpy.random.randint(0, 8, numpy.random.randint(1, 16)), float)
		assert numpy.allclose(overlap_add_convolve(a, b, fftconvolve), numpy.convolve(a, b))

def cumsum_via_reverse_sum(arr):  # Better cumsum for right-skewed distributions
	result = numpy.cumsum(arr[::-1])[::-1]
	s = result[0]
	result = s - result
	result = numpy.roll(result, -1)
	result[-1] = s
	return result

# inverse of cumsum()
def diff(arr, has_prepend='prepend' in (lambda code: code.co_varnames[:code.co_argcount])(numpy.diff.__code__)):
	if has_prepend:  # Only available in newer versions of NumPy
		result = numpy.diff(arr, prepend=0)
	else:
		result = numpy.diff(numpy.pad(arr, [(1, 0)], 'constant', constant_values=0))
	return result

class FixedTimeZone(tzinfo):
	def __init__(self, *args, **kwargs):
		self._utcoffset = kwargs.pop('utcoffset', None)
		self._dst = kwargs.pop('dst', None)
		self._name = kwargs.pop('name', None)
		super(FixedTimeZone, self).__init__(*args, **kwargs)
	def utcoffset(self, dt): return self._utcoffset
	def dst(self, dt): return self._dst
	def tzname(self, dt): return self._name


def datetime_parse_iso8601(s, keep_as_tuple=False, timezone_cache=None, pattern_match=re.compile("^(\\d\\d\\d\\d)-(\\d\\d?)-(\\d\\d?)T(\\d\\d?):(\\d\\d?):(\\d\\d?)(?:\\.(\\d*))?(?:Z?|([+\\-]?\\d\\d?)(?::?(\\d\\d))?)$").match):
	int_ = int
	m = pattern_match(s)
	if m is None: raise ValueError("Could not parse date/time: " + repr(s))
	g = m.groups()
	if g[7] is None:
		timezone = None
	else:
		tz0 = g[7]
		tz1 = g[8]
		tz_min = int(tz1)
		is_utc = False
		tz_key = (int(tz0), tz_min) if tz_min else tz0
		timezone = timezone_cache.get(tz_key) if timezone_cache is not None else None
		if timezone is None:
			timezone = FixedTimeZone(utcoffset=timedelta(hours=int_(tz0), minutes=0 if is_utc else tz_min))
			if timezone_cache is not None:
				timezone_cache[tz_key] = timezone
	args = (int_(g[0]), int_(g[1]), int_(g[2]), int_(g[3]), int_(g[4]), int_(g[5]), int_(g[6].ljust(6, "0")[:6]) if g[6] is not None else 0, timezone)
	return datetime(*args) if not keep_as_tuple else args

def epochtime(tuple_with_timezone):
	result = calendar.timegm(tuple_with_timezone)
	if len(tuple_with_timezone) > 7 and tuple_with_timezone[7] is not None:
		result += tuple_with_timezone[7].utcoffset(tuple_with_timezone)
	return result

def read_header_line(iterator):
	quoted = False
	remaining = None
	so_far = []
	stop = False
	while not stop:
		try: line = iterator.readline()
		except StopIteration: break
		to_append = line
		if remaining is None: remaining = line[:0]
		for i, c in enumerate(line):
			if c == '"': quoted = not quoted
			elif c in '\r\n':
				if not quoted:
					if line[i:i+1] == '\r': i += 1
					if line[i:i+1] == '\n': i += 1
					to_append = line[:i]
					remaining = line[i:]
					stop = True
					break
		so_far.append(to_append)
	return (so_far[0][:0].join(so_far) if len(so_far) > 0 else '', remaining)

def make_csv_reader(infile):
	(first_line, after_first_line) = read_header_line(infile)
	prefix = first_line + after_first_line + infile.read(1 << 12)
	dialect = csv.Sniffer().sniff(prefix)
	if dialect.escapechar is None and not dialect.doublequote: dialect.doublequote = True
	content = prefix + infile.read()
	result = None
	if "\"" not in content:
		lines = content.splitlines()
		result = map(lambda line: line.split(","), lines)
	else:
		result = csv.reader(content.splitlines(True), dialect)
	return result

Ticket_dtype = [('user_id', '<i4'), ('duration', '<i4'), ('split_days', '<i4')]

class Distribution(object):  # All instances are assumed to be independent! This means self - self != 0!
	_hash = None
	def __init__(self, distribution=(lambda a: a.setflags(write=False) and False or a)(numpy.asarray([1.0])), begin=0):
		# Does NOT check if the elements add up to one!
		distribution = numpy.asarray(distribution, float)
		nonzeros = distribution.nonzero()[0]
		if len(nonzeros) == 0:
			assert len(distribution) > 0
			nonzeros = numpy.append(nonzeros, 0)
		cumulative_complement = numpy.cumsum(distribution)
		cumulative_complement = numpy.subtract(1, cumulative_complement, cumulative_complement)
		jnonzero = nonzeros[-1].tolist() + 1 if len(nonzeros) > 0 else 0
		inonzero = nonzeros[0].tolist() if len(nonzeros) > 0 else jnonzero
		distribution.flags.writeable = False
		cumulative_complement.flags.writeable = False
		self.distribution = distribution[inonzero:jnonzero]
		self.cumulative_complement = cumulative_complement[inonzero:jnonzero]
		self.begin = begin + inonzero
		self.end = self.begin + (jnonzero - inonzero)
	@staticmethod
	def pad_shifted_array(arr, arr_begin, i, j, lval=0, rval=0, out=None):
		arr_end = arr_begin + len(arr)
		if i is None: i = arr_begin
		if j is None: j = arr_end
		v = arr[max(i - arr_begin, 0) : max(j - arr_begin, 0)]
		n = len(v)
		lpad = max(arr_begin - i, 0)
		rpad = max(j - arr_end, 0)
		if out is None:
			out = numpy.pad(v, [(lpad, rpad)], 'constant', constant_values=[(lval, rval)])
		else:
			begin = lpad + n
			out[0 : lpad] = lval
			out[lpad : begin] = v
			out[begin : begin + rpad] = rval
		return out
	def pdf_range(self, i=None, j=None, out=None):
		arr = self.distribution
		return Distribution.pad_shifted_array(self.distribution, self.begin, i, j, 0, 0, out)
	def cdf_complement_range(self, i=None, j=None, out=None):
		arr = self.cumulative_complement
		return Distribution.pad_shifted_array(arr, self.begin, i, j, 1, arr[-1], out)
	@staticmethod
	def chop(arr, cutoff):
		to_chop = arr <= cutoff
		denom = 1 - arr[to_chop].sum()
		arr[to_chop] = 0
		arr /= denom
	def plus(self, other, cutoff=None):
		assert other is not self
		dist = overlap_add_convolve(self.distribution, other.distribution, fftconvolve)
		if cutoff is not None: self.chop(dist, cutoff)
		return self.__class__(dist, self.begin + other.begin)
	def __repr__(self):
		if self.begin >= 0 and (len(self.distribution) < 0x10 or len(numpy.flatnonzero(self.distribution)) >= len(self.distribution) // 2):
			return repr([0.0] * self.begin + numpy.asarray(self.distribution).tolist())
		return repr(dict(filter(lambda pair: pair[1] != 0, map(lambda k, v: (k, v), range(self.begin, self.end), numpy.asarray(self.distribution).tolist())))) if True else "{begin: %s, distribution: %s}" % (repr(self.begin), repr(self.distribution))
	def __hash__(self):
		if self._hash is None:
			self._hash = hash((self.begin, self.distribution.tobytes()))
		return self._hash
	def __eq__(self, other): return self.begin == other.begin and numpy.array_equal(self.distribution, other.distribution)
	def __lt__(self, other):
		return self.begin < other.begin or not (other.begin < self.begin) and array_lexicographical_compare(self.distribution, other.distribution) < 0

class WaitTimePredictor(object):
	PROBABILITY_MAGNITUDE_CUTOFF = 1E-8
	DURATION_PERCENTILE_CUTOFF = 100 * (1 - 1.0 / 2000)  # Cut off data above this percentile
	MAX_SPLIT_DAY_DURATION_SECONDS = 0  # Max duration for tickets spanning different days (optional)
	min_duration_seconds = 35  # Min duration for any ticket
	max_duration_seconds = 16 * 60 * 60  # Max duration for any ticket
	bin_size = 1  # number of seconds to bin together
	bin_smoothing = (lambda a: numpy.asarray(a, float) / sum(a))([1])  # smoothing convolution filter (normalized)
	tickets = None
	scratch = None
	def __init__(self, **kwargs):
		for key, value in kwargs.items():
			if not hasattr(WaitTimePredictor, key):
				raise TypeError("invalid keyword argumen: " + str(key))
			setattr(self, key, value)
	def load(self, csv_reader):
		if self.tickets is None:
			self.tickets = []
		headers = None
		rows = None
		for row in csv_reader:
			if headers is None:
				if 'event_type' not in row: break
				headers = dict(map(lambda pair: (pair[1], pair[0]), enumerate(row)))
				rows = []
			else:
				rows.append(row)
		if rows is not None:
			id_header = headers['id']
			user_id_header = headers['user_id']
			ticket_id_header = headers['ticket_id']
			event_type_header = headers['event_type']
			time_header = headers['time']
			if False: rows.sort(key=lambda row: (row[time_header], row[id_header]))
			tickets_assigned = {}
			timezone_cache= {}
			for row in rows:
				event_type = row[event_type_header]
				if event_type == 'assign':
					tickets_assigned[row[ticket_id_header]] = row
				elif event_type == 'unassign':
					tickets_assigned.pop(row[ticket_id_header], None)
				elif event_type == 'resolve':
					prev_row = tickets_assigned.pop(row[ticket_id_header], None)
					if prev_row is not None:
						tassign = datetime_parse_iso8601(prev_row[time_header], True, timezone_cache)
						tresolve = datetime_parse_iso8601(row[time_header], True, timezone_cache)
						same_hour = tresolve[0] == tassign[0] and tresolve[1] == tassign[1] and tresolve[2] == tassign[2] and tresolve[3] == tassign[3]
						if self.MAX_SPLIT_DAY_DURATION_SECONDS is None or same_hour:
							if same_hour:
								duration = int(tresolve[4] - tassign[4]) * 60 + (tresolve[5] - tassign[5])
							else:
								duration = int((datetime(*tassign) - datetime(*tresolve)).total_seconds())
							split_days = 0
						else:
							tassign_epoch = epochtime(tassign)
							tresolve_epoch = epochtime(tresolve)
							duration = int(tresolve_epoch - tassign_epoch)
							split_days = (datetime.fromtimestamp(tresolve_epoch).date() - datetime.fromtimestamp(tassign_epoch).date()).days
						if duration <= 3600 and duration > 5:
							self.tickets.append((row[user_id_header], duration, split_days))
				elif event_type == 'create' or event_type == 'delete' or event_type == 'describe' or event_type == 'update_location':
					pass
				else:
					raise ValueError("Unknown event_type: " + event_type)
	def compute_wait_times(self):
		if self.tickets is None:
			raise ValueError("no ticket information loaded; can't compute distributions")
		filtered = numpy.asarray(self.tickets, dtype=Ticket_dtype)
		filtered = filtered[~(filtered['duration'] < self.min_duration_seconds)]
		filtered = filtered[~(filtered['duration'] > self.max_duration_seconds)]
		if self.MAX_SPLIT_DAY_DURATION_SECONDS is not None: filtered = filtered[~((filtered['split_days'] > 0) & (filtered['duration'] > self.MAX_SPLIT_DAY_DURATION_SECONDS))]
		filtered = filtered[~(filtered['duration'] > numpy.percentile(filtered['duration'], self.DURATION_PERCENTILE_CUTOFF))]
		return filtered['duration']
	def compute_wait_time_distribution(self, durations):
		# sys.stdout.write("Median:  %.0f minutes\n" % (numpy.median(durations).tolist() / 60,))
		# sys.stdout.write("Mean:    %.0f minutes\n" % (numpy.mean(durations).tolist() / 60,))
		# sys.stdout.write("Std dev: %.0f minutes\n" % (numpy.std(durations).tolist() / 60,))
		(dist, bins) = numpy.histogram(durations, numpy.arange(0, durations.max() + self.bin_size, self.bin_size), density=True)
		dist *= self.bin_size
		dist = numpy.convolve(dist, self.bin_smoothing, 'same')
		to_chop = dist <= self.PROBABILITY_MAGNITUDE_CUTOFF
		denom = 1 - dist[to_chop].sum()
		dist[to_chop] = 0
		dist[:] /= denom
		nonzeros = dist.nonzero()[0]
		return dist[:nonzeros.max() + 1 if len(nonzeros) > 0 else 0]
	def _calculate_help_time_probabilities(self, availabilities):
		# We want the probability that the i'th TA will help at time ti.
		# Probability of being helped by i'th TA at time ti is equal to
		#         the probability of NOT being helped before time ti
		#   times the probability that the i'th TA becomes ready at time ti
		#   divided by the total probability of ANY TAs becoming ready at time ti
		# Also note:  CDF(min_k(x[k])) = 1 - prod_k(1 - CDF(x[k]))
		# Sample table to work through:
		#    |  0 |  1  |  2
		# ---+----+-----+----
		#  A |  0 | 1   | 0
		#  B |  0 | 1/3 | 2/3
		#  C |  0 | 1/4 | 3/4
		# Another example:
		# A: [0, 3/4, 1/4]
		# B: [0, 2/3, 1/3]
		tminbegin = min(map(lambda v: v.begin, availabilities))
		tmaxend = max(map(lambda v: v.end, availabilities))
		scratch_shape = (2, len(availabilities), tmaxend - tminbegin)
		if self.scratch is not None and all(scratch_shape >= self.scratch):
			scratch = self.scratch
			self.scratch = None
		else:
			scratch = numpy.empty(scratch_shape, float)
			assert scratch.__setitem__(slice(None), numpy.nan) is None or True  # in debug mode, fill with NaNs to find any bugs
		(pdfs, cdf_complements) = (a, b) = scratch[tuple(map(slice, scratch_shape))]
		# WARNING: The code below is tricky! The buffers are re-used! Be VERY careful not to trash buffers that are used afterward!
		for i in range(len(pdfs)):
			availabilities[i].pdf_range(tminbegin, tmaxend, pdfs[i])
		for i in range(len(cdf_complements)):
			availabilities[i].cdf_complement_range(tminbegin, tmaxend, cdf_complements[i])
		with numpy.errstate(divide='ignore', invalid='ignore'):
			min_cdf = numpy.subtract(1, numpy.prod(cdf_complements, 0, out=b[0, :]), out=b[0, :])
			min_pdf = diff(min_cdf)
			prob_help = numpy.nan_to_num(numpy.multiply(min_pdf[numpy.newaxis, :],
				numpy.divide(pdfs, pdfs.sum(axis=0, out=None, keepdims=True), out=a),
				out=a), False)
		return (tminbegin, prob_help.sum(0), prob_help.sum(1), tmaxend)
	# TODO: Verify by simulation
	def get_wait_itimes(self, dist, instructor_start_times, queue_depth):
		ABSTOL_DECIMAL_PLACES = 12
		help_time = Distribution(dist, 0)
		start_time_discretized_distributions = list(map(lambda tistart: Distribution(numpy.asarray([1], float), tistart), numpy.divide(instructor_start_times, self.bin_size).astype(int).tolist()))
		memo = {}  # set to None to disable memoization (for correctness verification)
		def helper(*args):
			(nq, weight, ta_start_times) = args
			result = memo.get(args) if memo is not None else None
			if result is None:
				# Compute the wait time for the current state
				(tminbegin, time_help_probs, instructor_help_probs, tmaxend) = self._calculate_help_time_probabilities(ta_start_times)
				assert numpy.allclose(time_help_probs.sum(), 1, atol=1E-4), "probabilities don't add up to 1; they add up to " + repr(time_help_probs.sum().tolist())
				wait_itimes = Distribution(time_help_probs, tminbegin)
				result = []
				if nq < queue_depth:
					result.append((nq, 1.0, wait_itimes, list(ta_start_times)))
				# Compute the next possible queue state
				if nq < queue_depth - 1:
					prev_subargs = None; prev_subresults = None; prev_wi = None
					for i in range(len(instructor_start_times)):
						wi = instructor_help_probs[i].tolist()
						ta_end_times = ta_start_times[:i] + (ta_start_times[i].plus(help_time, self.PROBABILITY_MAGNITUDE_CUTOFF),) + ta_start_times[i + 1:]
						ta_end_times = sorted(ta_end_times)  # Because TAs are indistinguishable (order doesn't matter) -- this lets us memoize (and maybe prune to approximate later)
						subargs = (nq + 1, 1.0, tuple(ta_end_times))
						if memo is not None and prev_subargs == subargs:
							cached = memo[subargs]
							result[len(result) - len(prev_subresults):] = map(lambda item, prev: item[:1] + (prev[1] + wi * item[1],) + item[1+1:], prev_subresults, result[len(result) - len(prev_subresults):])
							prev_subresults = prev_subresults  # To denote that it hasn't changed
						else:
							prev_subresults = list(helper(*subargs))
							result.extend(map(lambda item: item[:1] + (wi * item[1],) + item[1+1:], prev_subresults))
						prev_subargs = subargs; prev_wi = wi
				result = list(result)
				result[:] = map(lambda item: item[:1] + (item[1] * weight,) + item[1+1:], result)
				result[:] = filter(lambda item: item[1] >= self.PROBABILITY_MAGNITUDE_CUTOFF, result)
				if memo is not None:
					memo[args] = tuple(result)
			return result
		return helper(0, 1.0, tuple(start_time_discretized_distributions))

def sparse_to_dense_pmf(weights, normalize=False):
	i = min(min(weights.keys()), 0)
	j = max(weights.keys()) + 1
	if i < 0: raise ValueError("start index cannot be below zero")
	w = numpy.zeros(j - i, float)
	for k, v in weights.items():
		w[k] = v
	if normalize:
		w /= numpy.sum(w)
	return w


def avgWaitTimeList(*args):
	instructor_start_times = [0] * int(args[2])
	queue_depth = len(instructor_start_times) + int(args[1])
	percentile = 0.50
	bin_size = 1
	DEBUG = False
	logging = DEBUG
	predictor = WaitTimePredictor(bin_size=bin_size)
	if DEBUG:
		samples_to_generate = 10000
		wait_time_dist = numpy.asarray(sparse_to_dense_pmf({1 * 60: 0.5, 3 * 60: 0.5}, True), float)
		waiting_times = numpy.random.choice(len(wait_time_dist), samples_to_generate, True, y / numpy.sum(wait_time_dist))
	else:
		nloaded = 0
		tstart = default_timer()
		for filename in args[0:1]:
			if os.path.splitext(filename)[1].lower() == ".csv":
				with open(filename, "r") as infile:
					predictor.load(make_csv_reader(infile))
					nloaded += 1
		tend = default_timer()
		# if nloaded > 0:
		# 	# sys.stderr.write("Loading data took %.2f seconds\n" % (tend - tstart,))
		waiting_times = predictor.compute_wait_times()
		wait_time_dist = predictor.compute_wait_time_distribution(waiting_times)
	# sys.stdout.write("Resolution (bin size): %s second(s)\n" % (bin_size,))
	if queue_depth <= len(instructor_start_times): sys.stderr.write("NOTE: Fewer people on queue than available instructors; everyone will be serviced immediately.\n")
	tstart = default_timer()
	queue_wait_itimes = []
	for (depth, weight, wait_itimes, ta_start_itimes) in predictor.get_wait_itimes(wait_time_dist, instructor_start_times, queue_depth):
		while depth >= len(queue_wait_itimes): queue_wait_itimes.append([])
		queue_wait_itimes[depth].append((weight, wait_itimes))
		# if logging: sys.stdout.write("%s\n" % ("\t".join(map(str, [weight, wait_itimes, ta_start_itimes])),))
	tend = default_timer()
	# if tend - tstart >= 0.05: sys.stderr.write("Calculating queue wait times took %.2f seconds\n" % (tend - tstart,))
	# sys.stdout.write("Wait times <= [%s] min(s) with %.0f%% probability for %d instructors starting at T = %s min(s)\n" % (
	# 	", ".join(map(
	# 		lambda t: "%.2g" % (t / 60.0,),
	# 		map(
	# 			lambda queue_k_wait_times: sum(map(
	# 				lambda info: info[0] * bin_size * numpy.searchsorted(numpy.subtract(1, info[1].cdf_complement_range(0)), percentile, 'left'),
	# 				queue_k_wait_times)),
	# 			queue_wait_itimes))),
	# 	percentile * 100,
	# 	len(instructor_start_times),
	# 	list(map(lambda t: t / 60.0, instructor_start_times))
	# ))
	# smoothing_window = numpy.median(waiting_times) / 5
	# sys.stdout.write("Plotting with a smoothing window of %s; please wait as smoothing may be slow...\n" % (smoothing_window,))
	lw = 96.0 / 120
	usetex = False
	(rows, cols) = (2, 1)
	# import matplotlib, matplotlib.pyplot
	# matplotlib.rc('text', usetex=usetex)
	# matplotlib.rc('font', family='serif', serif=['cmr10' if usetex else 'Latin Modern Roman'], size=11)
	# pyplot = matplotlib.pyplot
	# from matplotlib import pyplot
	# fig = pyplot.figure(2, (9.0 * cols, 3.5 * rows), 120, None, None, False) or matplotlib.figure.Figure()
	xmax = None
	# if True:
	# 	xcutoff = 60 * 60
	# 	import scipy.stats
	# 	dist = scipy.stats.gengamma
	# 	x = numpy.arange(len(wait_time_dist))[:xcutoff]
	# 	x_displayed = x / 60.0
	# 	xmin = 0.0
	# 	y0 = dist.pdf(x, *dist.fit(waiting_times[waiting_times <= xcutoff], floc=xmin))
	# 	y1 = scipy.stats.gaussian_kde(waiting_times, smoothing_window / float(len(x)))(x); y1 /= numpy.sum(y1)
	# 	y2 = wait_time_dist[:xcutoff]; y2 = y2 / numpy.sum(y2)
		# axes = fig.add_subplot(rows, cols, 1) or matplotlib.axes.Axes()
		# artists_and_legends = []
		# axes.set_title("Waiting time of a single person (minutes)")
		# # axes.grid(True, which='major')
		# artists_and_legends.append((axes.plot(x_displayed, y0, lw=lw, color='black')[0], "Estimated %s distribution (ideal)" % (dist.name,)))
		# artists_and_legends.append((axes.fill_between(x_displayed, y1, lw=lw), "Smoothed empirical distribution"))
		# artists_and_legends.append((axes.fill_between(x_displayed, y2, lw=lw), "Empirical distribution"))
		# axes.legend(*zip(*artists_and_legends[::-1]))
		# axes.set_xlim(xmin, x_displayed.max())
		# axes.set_ylim(0, numpy.max((y0.max(), y1.max(), y2.max())))
		# xmax = axes.get_xlim()[1]
		# axes.get_xaxis().set_major_locator(matplotlib.ticker.MultipleLocator(5))
		# axes.get_xaxis().set_minor_locator(matplotlib.ticker.MultipleLocator(1))
	if True:
		# axes = fig.add_subplot(rows, cols, 2) or matplotlib.axes.Axes()
		# axes.set_title("Waiting times of the people waiting on the queue (minutes)")
		maxend = max(map(lambda queue_k_wait_itimes: max(map(lambda info: info[1].end, queue_k_wait_itimes)), queue_wait_itimes))
		artists_and_legends = []
		avgTimeList, stdDevList = [], []
		for k, queue_k_wait_itimes in list(enumerate(queue_wait_itimes[len(instructor_start_times):]))[::-1]:
			dist = numpy.sum(list(map(lambda info: info[0] * info[1].pdf_range(0, maxend), queue_k_wait_itimes)), 0)
			cumdist = numpy.cumsum(dist)
			ibegin = 0
			plot_cumulative = False
			iend = len(cumdist) - (0 if plot_cumulative else (cumdist[::-1] < (1 - 1.0 / (1 << 8))).argmax())
			dist_to_plot = cumdist if plot_cumulative else dist
			dist_to_plot = dist_to_plot[ibegin:iend]
			x = numpy.linspace(ibegin * bin_size, iend * bin_size, len(dist_to_plot), False)
			x_displayed = x / 60
			# artists_and_legends.append((axes.fill_between(x_displayed, dist_to_plot, lw=lw), "Waiting time of person #%s on queue" % (k + 1,)))
			# axes.axvline(x=numpy.average(x_displayed, None, dist_to_plot / dist_to_plot.sum()), lw=lw, color='black')
			avgTimeList.append(numpy.average(x_displayed, None, dist_to_plot / dist_to_plot.sum()))
			stdDevList.append(numpy.std(x_displayed, None))

		# axes.legend(*zip(*artists_and_legends[::-1]))
		# axes.set_xlim(0, max(xmax or 0, axes.get_xlim()[1]))
		# axes.set_ylim(0, axes.get_ylim()[1])
		# axes.get_xaxis().set_major_locator(matplotlib.ticker.MultipleLocator(5))
		# axes.get_xaxis().set_minor_locator(matplotlib.ticker.MultipleLocator(1))
	#fig.tight_layout()
	#if False: fig.savefig('waiting-time.png', transparent=True)
	#pyplot.show(fig)
	return avgTimeList, stdDevList
