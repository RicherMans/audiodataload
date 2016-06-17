# Audioload

Audioload is a collection of Torch dataset loaders used for audio processing. Compared to other libraries ( [elements dataload](https://github.com/Element-Research/dataload)
), this library aims for high performance while still being scalable with large datasets. That said, this library only loads data on the fly, as much as it needs, thus it does not require a large amount of memory, which makes it possible to work with very big datasets.
Following classes encompass the library:

* [BaseDataloader](#adl.basedataloader) : The base abstract class that should never be directly initiated.
* [WaveDataloader](#adl.wavedataloader) : A dataloader for raw wave files ( useful for any type of feature extraction with CNN's )
* [Sequenceiterator](#adl.seqenceiterator) : Wraps a given [dataloader](#adl.basedataloader) to return sequences of samples instead of only single samples. This iterator should be used when using recurrent networks.
* [Htkiterator](#adl.htkiterator): A dataloader for [HTK](http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node58.html#SECTION03271000000000000000) formatted features.
* [Hdf5iterator](#adl.hdfiterator): Wraps the given [dataloader](#adl.basedataloader) with this iterator to not use it's internal loading methods, but uses a preprocessed HDF5 file instead.
* [Asynciterator](#adl.asynciterator): Wraps the given [dataloader](#adl.basedataloader) with an iterator that allows multithreaded processing. This can be beneficial and allows to speed up the loading process, but also uses up a lot more memory.

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

### [n] usize()

Returns the number of utterances in the current dataset

### [n] dim()

Returns the datadimension as a number.

### [n] nClasses()

Returns the number of target classes


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
 * `filepaths` : a table which contains the filepaths of the loaded utterances. Useful if one wants to extract features from the utterances.

The iterator will return batches of `inputs` and `targets` of size at most `batchsize` until `epochsize` samples have been returned.

When this method is called, the subclass methods ```getSample``` method is explicitly called and needs to return at least a list of the inputs and labels for the current batch. The implementation does differ from class to class, thus please refer to the wavedataloader#uttiterator for further information.


<a name='adl.basedataloader.sampleiterator'></a>
### [iterator] sampleiterator(batchsize [n], epochsize [n])

Returns an iterator on the samples of the . By default the iterator returns the samples of the given utterances in order, thus a single batch will (approximately) contain around 1 single target. For tasks where no sequence training is done, this is the preferred method to train and evaluate the model.


<a name='adl.wavedataloader'></a>
## WaveDataloader

WaveDataloader loads features as raw waves from the disk. Features can be accessed utterancewise or samplewise.

### init( filepath, framesize [n] , [ shift [n] ])

* `filepath`: The filepath specifying the feature file containing all the features and corresponding labels.
* `framesize` : The size for one frame. The initial size into which an utterance is splitted to. E.g. of we have a wave utterance which consists of 1s 16k Hz audio, we obtain `16000` samples. Specifying `framesize` to be 400, we would obtain `16000/400 = 40` different frames. Generally to obtan the number of frames we calculate:
`floor((audiosize - framesize)/shift + 1)`, whereas `audiosize` is the length of the audiofile and `shift` is the parameter below ( defaults to `framesize`)
* `shift` : The frameshift between two frames, which enables overlapping between adjacent frames. The default is `framesize` thus no overlapping.
* `usemaxseqlength` is per default set to zero. This flag sets the `seqlen` to the maximum seqlength for the current dataset. It makes it possible to use [uttiterator](#adl.wavedataloader.uttiterator) with some fixed size of batches. It implicitly pads all other arrays according to the `padding` option. This is useful for evaluation if whole utterances are to be inputted and batchnormalization is needed.


<a name='adl.wavedataloader.uttiterator'></a>
### [iterator] uttiterator(batchsize [n], epochsize [n])

Returns an iterator to the utterances. The returned utterances are always of size `nsample * datadim`. `nsample` is the number of samples for each utterance, thus this might wary.

* `batchsize` : This is set to `1` and cannot be used with any other value for `batchsize`. If iterating over batches of utterances is needed, please refer to [Sequenceiterator](#adl.seqenceiterator).
* `epochsize` : The size of the epoch, via default its `self:usize()` thus being evaluated over all the utterances.

Example to iterate with 1 batchsize ( useful for evaluation of sequences ):

```lua
local traindatawave = audiodataload.WaveDataloader{
    path = 'traindata',
    framesize = 100,
}
-- defaults to batchsize 1, thus returning nsample * datadim
local sampleiterator = traindatawave:uttiterator()
for done, max, input,target, filepath in sampleiterator do
    -- Do stuff
end
```

<a name='adl.wavedataloader.sampleiterator'></a>
### [iterator] sampleiterator(batchsize [n], epochsize [n], randomize [false])

Returns an iterator to single samples of the wavedataset. The output is at most of size `batchsize` and `epochsize` defaults to the dataset's samplesize. If `randomize` is set to be true, it randomizes the input for each iteration.

**Note** that this iterator always loads in a full utterance and then cut's it appropriately, meaning that if utterances are very large, this iterator can be very slow. In this case please wrap it with [Hdf5iterator](#adl.hdfiterator).

Example:

```lua
local traindatawave = audiodataload.WaveDataloader{
    path = 'traindata',
    framesize = 100,
    seqlen = 10,
}
-- Randomize the input
-- batchsize 128
-- nil lets the iteration use the full dataset as epochsize
local sampleiterator = traindatawave:sampleiterator(128,nil,true)
for done, max, input,target, filepath in sampleiterator do
    -- Do stuff
end
```

<a name='adl.htkdataloader'></a>

## Htkdataloader

A dataloader specifically for HTK extracted features. The dataloader ignores all the extra parameters of the HTK feature and simply loads the data.

### init( filepath )

`filepath`: The filepath specifying the feature file containing all the features and corresponding labels.

<a name='adl.htkdataloader.sampleiterator'></a>
### [iterator] sampleiterator(batchsize [n], epochsize [n], randomize [false])

Returns an iterator to single samples of the wavedataset. The output is at most of size `batchsize` and `epochsize` defaults to the dataset's samplesize. If `randomize` is set to be true, it randomizes the input for each iteration.

The parameters behave exactly as in the [basedataloader's](#adl.adl.basedataloader.sampleiterator).

<a name='adl.htkdataloader.uttiterator'></a>

### [iterator] uttiterator(batchsize [n], epochsize [n])

Returns an iterator to the utterances. The returned utterances are always of size `nsample * datadim`. `nsample` is the number of samples for each utterance, thus this might wary.

The parameters behave exactly as in the [basedataloader's](#adl.adl.basedataloader.uttiterator).


<a name='adl.seqenceiterator'></a>

## Sequenceiterator


This class wraps a [dataloader](#adl.BaseDataloader) to iterate over batches of sequences.

### init( module , [ sequencelength [n] ], [ usemaxseqlength [false]], [padding [left]] )

* `module` : Specifies the module to wrap sequence iteration over.
* `sequencelength` : Per default 1, this specifies the sequence length of each returned sample. If data cannot be fitted into the sequence (e.g. there is not enough available), sequences will zeroed out.
* `usemaxseqlength` : If this argument is set to true, we ignore the `sequencelength` argument and automatically set the sequence length to the largest sequence in the dataset( the given model).
* `padding` : In some cases the given `seqlen` is larger than an utterance, thus producing an overly large inputtensor. In this cases we pad the returned tensor with zeros. This parameter specifies if the padding is done on the left ( e.g. `0000011111`) or on the right (e.g. `11111100000`).

<a name='adl.sequenceiterator.sampleiterator'></a>
### [iterator] sampleiterator(batchsize [n], epochsize [n], randomize [false])
Performs the wrapped module's iteration, but returns instead of samples of size `batchsize * datadim`, samples of size `seqlength * batchsize * datadim`. All seqences are stored and returned contiguously.
```lua
local filepath = "train.lst"
local framesize = 100
local dataloader = audioload.WaveDataloader(filepath,framesize)
local seqiter = audioload.Sequenceiterator{module=dataloader,usemaxseqlength =true}
for start,all,input,target in seqiter:sampleiterator(128,nil,true) do

end
```

<a name='adl.sequenceiterator.uttiterator'></a>
### [iterator] uttierator(batchsize [n], epochsize [n])

Returns a tensor of size `seqlength * batchsize * datadim`. If `seqlength` is smaller than any of the utternaces seqencelength, the tensor will be cropped. To avoid cropping, pass `usemaxseqlength` to the constructor.

```lua
local filepath = "train.lst"
local seqlenth = 50
local framesize = 100
local dataloader = audioload.WaveDataloader(filepath,framesize)
local seqiter = audioload.Sequenceiterator(dataloader,seqlenth)
for start,all,input,target in seqiter:uttiterator(128,nil,true)do
    --- Do the iteration
end

```


<a name='adl.hdfiterator'></a>
## HDF5Iterator

This class is a wrapper, most likely to be used with [WaveDataloader](#adl.wavedataloader), because it provides a significant performance boost for sampleiteration.

### init( module , filepath, [ chunksize [n]] )

* `module` : Specifies the module to wrap hdf5 iteration over.
* `filepath` : The file where an hdf5 file will be dumped to. If the file exists it does not create a new file. Please be reminded that it assumes a flattened tensor as an input. Best practice is to firstly dump a new hdf5 file, not manually creating one ( so that the dimensions match ).
* `chunksize` : If this argument is set to some number, deflate is used to compress the hdf5 file. With deflate, we split the data into chunks of `chunksize`. More information can be found [here](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md)

<a name='adl.hdfiterator.sampleiterator'></a>
### [iterator] sampleiterator(batchsize [n], epochsize [n], randomize [false])
Performs the wrapped module's iteration, but loads the data from the hdf5 file using `partial` loading.

<a name='adl.hdfiterator.uttiterator'></a>
### [iterator] uttiterator(batchsize [n], epochsize [n], randomize [false])
Performs the wrapped module's iteration, but loads the data from the hdf5 file using `all` loading.

<a name='adl.asynciterator'></a>
## AsyncIterator

This class wraps a given dataload to return consequent samples

### init (module,[ threads [n], serialmode [string]])

Inits the asyncdataloader , where ```module``` needs to be a valid subclass of [basedataloader](#adl.basedataloader). ```serialmode``` specifies the dataformat of the wrapped module which is dumped onto the filesystem. By default this is ``binary``, but can also be changed to ``ascii``.

### [iterator] sampleiterator(batchsize [n], epochsize [n], randomize [false])

Returns samples from the given wrapped around module, but uses a multithreaded execution scheme. The order of returned samples is not guaranteed to be the same as in non multithread iteration.

### [iterator] uttiterator(batchsize [n], epochsize [n], randomize [false])

Not yet implemented ( it is also relatively useless)
