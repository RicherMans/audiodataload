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


### [n] size()

### [n] dim()

### [n] nClasses()

### random()

### [iterator] uttiterator(batchsize [n], epochsize [n])

When this method is called, the subclass methods ```getSample``` method is explicitly called and needs to return at least a list of the inputs and labels for the current batch. The implementation does differ from class to class, thus please refer to the wavedataloader#uttiterator for further information.



### [iterator] sampleiterator(batchsize [n], epochsize [n])






<a name='wavedataloader'></a>
## WaveDataloader

WaveDataloader loads features from the

### [iterator] uttiterator(batchsize [n], epochsize [n])

Returns an iterator to the utterances itself. It has two different modes, first if ```batchsize == 1``` then it returns a 2d tensor, having dimensions ``` nsample

<a name='hdf5iterator'></a>
## HDF5Iterator
