import os
import pickle
import sys
from typing import Dict

from aesara.configdefaults import config


DISPLAY_DUPLICATE_KEYS = False
DISPLAY_MOST_FREQUENT_DUPLICATE_CCODE = False

dirs = []
if len(sys.argv) > 1:
    for compiledir in sys.argv[1:]:
        dirs.extend([os.path.join(compiledir, d) for d in os.listdir(compiledir)])
else:
    dirs = os.listdir(config.compiledir)
    dirs = [os.path.join(config.compiledir, d) for d in dirs]
keys: Dict = {}  # key -> nb seen
mods: Dict = {}
for dir in dirs:

    key = None
    try:
        with open(os.path.join(dir, "key.pkl")) as f:
            key = f.read()
        keys.setdefault(key, 0)
        keys[key] += 1
        del f
    except OSError:
        # print dir, "don't have a key.pkl file"
        pass
    try:
        path = os.path.join(dir, "mod.cpp")
        if not os.path.exists(path):
            path = os.path.join(dir, "mod.cu")
        with open(path) as f:
            mod = f.read()
        mods.setdefault(mod, ())
        mods[mod] += (key,)
        del mod
        del f
        del path
    except OSError:
        print(dir, "don't have a mod.{cpp,cu} file")

if DISPLAY_DUPLICATE_KEYS:
    for k, v in keys.items():
        if v > 1:
            print("Duplicate key (%i copies): %s" % (v, pickle.loads(k)))

nbs_keys: Dict = {}  # nb seen -> now many key
for val in keys.values():
    nbs_keys.setdefault(val, 0)
    nbs_keys[val] += 1

nbs_mod: Dict = {}  # nb seen -> how many key
nbs_mod_to_key = {}  # nb seen -> keys
more_than_one = 0
for mod, kk in mods.items():
    val = len(kk)
    nbs_mod.setdefault(val, 0)
    nbs_mod[val] += 1
    if val > 1:
        more_than_one += 1
    nbs_mod_to_key[val] = kk

if DISPLAY_MOST_FREQUENT_DUPLICATE_CCODE:
    m = max(nbs_mod.keys())
    print("The keys associated to the mod.{cpp,cu} with the most number of copy:")
    for kk in nbs_mod_to_key[m]:
        kk = pickle.loads(kk)
        print(kk)

print("key.pkl histograph")
l = list(nbs_keys.items())
l.sort()
print(l)

print("mod.{cpp,cu} histogram")
l = list(nbs_mod.items())
l.sort()
print(l)

total = sum(len(k) for k in list(mods.values()))
uniq = len(mods)
useless = total - uniq
print("mod.{cpp,cu} total:", total)
print("mod.{cpp,cu} uniq:", uniq)
print("mod.{cpp,cu} with more than 1 copy:", more_than_one)
print("mod.{cpp,cu} useless:", useless, float(useless) / total * 100, "%")

print("nb directory", len(dirs))
