import collections

d = collections.OrderedDict()
d['x'] = 100
d['y'] = 200
d['z'] = 300

for k, v in d.items():
    print(k, v)

d = {}
d['x'] = 100
d['y'] = 200
d['z'] = 300

for k, v in d.items():
    print(k, v)