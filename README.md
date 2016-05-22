# Audioload

Audioload is a collection of Torch dataset loaders used for audio processing. Other dataset libraries such as [elements dataload](https://github.com/Element-Research/dataload)
) have major disadvantages (mainly being not loading data on the fly), especially they aren't suited for speech related tasks.
Following classes encompass the library:

* BaseDataloader(#basedataloader)
* Hdf5iterator(#hdfiterator): Wraps the given dataloader with this iterator to not use it's internal loading methods, but uses a preprocessed HDF5 file instead
* WaveDataloader(#wavedataloader)

```lua
local audioload = require 'audioload'
```

<a name='basedataloader'></a>
## BaseDataLoader

BaseDataloader is the abstract main class of all the dataloader facilities. It provides two basic functions which are #sampleiterator and #uttiterator to iterate over sample or utterances respectively


### Methods



<a name='wavedataloader'></a>
## WaveDataloader

WaveDataloader loads features from the

<a name='hdf5iterator'></a>
## HDF5Iterator
