Exact Nearest Neighbor Search
=============================
[![Build Status](https://travis-ci.org/xinyandai/exactNNS.svg?branch=master)](https://travis-ci.org/xinyandai/exactNNS)


## Algorithm Description
1. build indexes for buckets with PQ
2. determine bucket's probe sequence, which could be lazy determined, and probe sequence is determined by
```
    d(q,c) = max( | ||q-c|| - ||c-x|| |) where c(x) = c
    may be wrong when ||q-c|| - ||c-x|| < 0
```
3. query
```
   while lower-bound < upper-bound && has next bucket:
        probe next bucket
        update upper-bound with current top K
        update lower-bound with current bucket center
```

## ENN without Inverted Multi Index
## ENN with Inverted Multi Index
