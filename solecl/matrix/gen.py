# -*- coding: utf-8 -*-

import random

random.seed()

nlist = [10000]

for n in nlist:
	M = n*10
	f = open('matrix%d.txt' % n, 'w')
	for i in xrange(n):
		s = ''
		d = random.random()*M # Диагональный элемент (для соблюдения условия диагонального преобладания)
		r = d
		for j in xrange(n):
			if(i == j):
				s += '%f ' % d
			else:
				t = random.random()*r/(n-j)
				r -= t
				s += '%f ' % t
		s += '%f\n' % (random.random()*M)
		f.write(s)
	f.close()
	print(n)