# Audioload

Audioload is a collection of Torch dataset loaders used for audio processing. Compared to other libraries ( [elements dataload](https://github.com/Element-Research/dataload)
), this library aims for high performance while still being scalable with large datasets. That said, this library only loads data on the fly, as much as it needs, thus it does not require a large amount of memory, which makes it possible to work with very big datasets.
Following classes encompass the library:

* [BaseDataloader](#adl.basedataloader) : The base abstract class that should never be directly initiated.
* [WaveDataloader](#adl.wavedataloader) : A dataloader for raw wave files ( useful for any type of feature extraction with CNN's )
* [Hdf5iterator](#adl.hdfiterator): Wraps the given dataloader with this iterator to not use it's internal loading methods, but uses a preprocessed HDF5 file instead.

```lua
local audioload = require 'audioload'
```

## Requirements

This library requires the following packages:

 * [torch-7](http://torch.ch/) as the main framework for all the computation
 * For the WaveDataloader we need the torch audio library [audio](https://github.com/soumith/lua---audio). If luarocks are used, it can be installed by running `luarocks install audio`
 * [torch-hdf5](https://github.com/deepmind/torch-hdf5) (not the lua hdf5 library) for wrapping a dataloader with this iterator

## Installation

The easiest way to install this library is to run

```bash
luarocks install audiodataload
```

## Data preparation

Since the data needs to be passed in a comprehensive file, the fileformat consists of datapath and the target number. Currently the targets need to be preenumerated to be in range `1..n`, otherwise an error will be thrown. For example:

```
/path/to/speaker1_utterance1.wav 1
/path/to/speaker1_utterance2.wav 1
/path/to/speaker1_utterance3.wav 1
/path/to/speaker2_utterance1.wav 2
/path/to/speaker3_utterance1.wav 3
```


<a name='adl.basedataloader'></a>
## BaseDataLoader

BaseDataloader is the abstract main class of all the dataloader facilities. It provides two basic functions which are [sampleiterator](#adl.basedataloader.sampleiterator) and [uttierator](#adl.basedataloader.uttierator) to iterate over sample or utterances respectively


### [n] size()

Returns the size (samplesize) of the current dataset

### [n] dim()

Returns the datadimension as a number.

### [n] nClasses()

Returns the number of target classes

### random ()

Internally randomizes the inputs and targets. This function should be used before the iterator is initalized e.g.
```lua
local traindatawave = audiodataload.WaveDataloader{
    path = 'traindata',
    framesize = 100,
}
-- Randomize the input first
traindatawave:random()
-- batchsize 16
local sampleiterator = traindatawave:sampleiterator(16)
for done, max, input,target, filepath in sampleiterator do
    -- Do stuff
end
```

<a name='adl.basedataloader.uttiterator'></a>
### [iterator] uttiterator(batchsize [n], epochsize [n])

Returns an iterator over a whole utterance. The two arguments are:

 * `batchsize` : a number that defines the returned input and target tensors batchsize.
 * `epochsize` : the number that defines how large the epoch is. The default is the overall size of the dataset.

Each iteration returns 4 values :

 * `k` : the number of samples processed so far. Each iteration returns a maximum of `batchsize` samples.
 * `n` : the number of overall utterances. This value is constant throughout the iteration.
 * `inputs` : a tensor containing a maximum of `batchsize` inputs.
 * `targets` : a tensor (or nested table thereof) containing targets for the commensurate inputs.

The iterator will return batches of `inputs` and `targets` of size at most `batchsize` until `epochsize` samples have been returned.

When this method is called, the subclass methods ```getSample``` method is explicitly called and needs to return at least a list of the inputs and labels for the current batch. The implementation does differ from class to class, thus please refer to the wavedataloader#uttiterator for further information.


<a name='adl.basedataloader.sampleiterator'></a>
### [iterator] sampleiterator(batchsize [n], epochsize [n])

Returns an iterator on the samples of the . By default the iterator returns the samples of the given utterances in order, thus a single batch will (approximately) contain around 1 single target. For tasks where no sequence training is done, this is the preferred method to train and evaluate the model.


<a name='adl.wavedataloader'></a>
## WaveDataloader

WaveDataloader loads features as raw waves from the disk. Features can be accessed utterancewise or samplewise.

### init(filepath, framesize [n] , [ shift [n] ], [ seqlen [ n ] ] , [ padding [n]] )

* `framesize` : The size for one frame. The initial size into which an utterance is splitted to. E.g. of we have a wave utterance which consists of 1s 16k Hz audio, we obtain `16000` samples. Specifying `framesize` to be 400, we would obtain `16000/400 = 40` different frames. Generally to obtan the number of frames we calculate:
`floor((audiosize - framesize)/shift + 1)`, whereas `audiosize` is the length of the audiofile and `shift` is the parameter below ( defaults to `framesize`)
* `shift` : The frameshift between two frames, which enables overlapping between adjacent frames. The default is `framesize` thus no overlapping.
* `seqlen` : If this is specified and greater than zero, we assume that one frame and following `seqlen` frames of size `framesize` are considered one sequence. Thus
* `padding` : In some cases the given `seqlen` is larger than an utterance, thus producing an overly large inputtensor. In this cases we pad the returned tensor with zeros. This parameter specifies if the padding is done on the left ( e.g. `0000011111`) or on the right (e.g. `11111100000`).




<a name='adl.wavedataloader.uttiterator'></a>
### [iterator] uttiterator(batchsize [n], epochsize [n])

Returns an iterator to the utterances itself. It has two different modes:
1. If `batchsize == 1` then it returns a 2d tensor, having dimensions `nsample * datadim`
2. If `batchsize > 1` then one must specify `seqlen > 1`. It returns a tensor of size `seqlen * batchsize * datadim`, where the `seqlen` does determinate how many sequences are taken into account, thus it either cuts off the input wave or extends them with zeros.

Example to iterate with 1 batchsize ( useful for evaluation of seqences ):

```lua
local traindatawave = audiodataload.WaveDataloader{
    path = 'traindata',
    framesize = 100,
}
-- Randomize the input first
traindatawave:random()
-- defaults to batchsize 1, thus returning nsample * datadim
local sampleiterator = traindatawave:uttiterator()
for done, max, input,target, filepath in sampleiterator do
    -- Do stuff
end
```

Example for iteration with seqlength and batchsize.

```lua
local traindatawave = audiodataload.WaveDataloader{
    path = 'traindata',
    framesize = 100,
    seqlen = 200,
}
traindatawave:random()
-- returning seqlen * nsample * datadim
-- it pads the tensors to have seqlen 200 with zeros from the left
local sampleiterator = traindatawave:uttiterator(128)
for done, max, input,target, filepath in sampleiterator do
    -- Do stuff
    print(input:size())-- has size seqlen (200) * batchdim(128) * datadim(100)
end
```

<a name='adl.wavedataloader.sampleiterator'></a>
### [iterator] sampleiterator(batchsize [n], epochsize [n])

Returns an iterator to single samples of the wavedataset. The output is at most of size `batchsize` and `epochsize` defaults to the dataset's samplesize.

**Note** that this iterator always loads in a full utterance and then cut's it appropriately, meaning that if utterances are very large, this iterator can be very slow. In this case please wrap it with [Hdf5iterator](#adl.hdfiterator).

Example:

```lua
local traindatawave = audiodataload.WaveDataloader{
    path = 'traindata',
    framesize = 100,
    seqlen = 10,
}
-- Randomize the input first
-- batchsize 128
local sampleiterator = traindatawave:sampleiterator(128)
for done, max, input,target, filepath in sampleiterator do
    -- Do stuff
end
```


<a name='adl.hdfiterator'></a>
## HDF5Iterator

This class is a wrapper, most likely to be used with [WaveDataloader](#adl.wavedataloader), because it provides a significant performance boost for sampleiteration.

### init( module , filepath, [ chunksize [n]] )

* `module` : Specifies the module to wrap hdf5 iteration over.
* `filepath` : The file where an hdf5 file will be dumped to. If the file exists it does not create a new file. Please be reminded that it assumes a flattened tensor as an input. Bestpractice is to firstly dump a new hdf5 file, not manually creating one ( so that the dimensions match ).
* `chunksize` : If this argument is set to some number, deflate is used to compress the hdf5 file. With deflate, we split the data into chunks of `chunksize`. More information can be found [here](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md)

<a name='adl.hdfiterator.sampleiterator'></a>
### [iterator] sampleiterator(batchsize [n], epochsize [n])
Performs the wrapped module's iteration, but loads the data from the hdf5 file using `partial` loading.

<a name='adl.hdfiterator.uttiterator'></a>
### [iterator] uttiterator(batchsize [n], epochsize [n])
Performs the wrapped module's iteration, but loads the data from the hdf5 file using `all` loading.
