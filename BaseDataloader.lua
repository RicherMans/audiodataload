local adl = require 'audiodataload._base'
local BaseDataloader = torch.class('adl.BaseDataloader', adl)


-- Reads a filelabel from the filelabels chartensor
function readfilelabel(labels)
    return ffi.string(torch.data(labels))
end
-- Number of utterances in the dataset
function BaseDataloader:usize()
    return self._uttsize
end

-- Returns the sample size of the dataset
function BaseDataloader:size()
    if not self.sampletofeatid and not self.sampletoclassrange then
        self.sampletofeatid,self.sampletoclassrange = self:sampletofeat(self:_getsamplelengths())
    end
    return self._nsamples
end

-- Returns the data dimensions
function BaseDataloader:dim()
    error "Not Implemented"
end

-- returns number of classes in the dataset
function BaseDataloader:nClasses()
    return self._numbertargets
end

-- These two methods should be called within the getSample/getutterance methods
-- to load a single sample/utterance out
-- Loads a single audio file into the memory. Used to overload by other classes and should be called during getSample()
function BaseDataloader:loadAudioSample(audiofilepath,...)
    error "Not Implemented"
end

function BaseDataloader:loadAudioUtterance(audiofilepath,...)
    error "Not Implemented"
end

-- templates to callback the iterators
function BaseDataloader:beforeIter(...)
    return
end
-- Callback for any tasks after finishing the iteration e.g. closing files
function BaseDataloader:afterIter(...)
    return
end

-- Function which needs to be overwritten which returns for the given filepath the number of samples available
function BaseDataloader:getNumAudioSamples(fname)
    error "Not Implemented"
end

local basecheck = argcheck{
    {
        name="path",
        type="string",
        help="Input file which contains the feature paths",
        check=function(filename)
            if paths.filep(filename) then return true else return false end
        end
    },
}

function BaseDataloader:__init(...)
    local filename = basecheck(...)

    local filelabels,targets = self:_readfilename(filename)

    -- Set the filelabels and targets for the subclasses
    self.filelabels = filelabels
    self.targets = targets

    -- set usize()
    self._uttsize = self.filelabels:size(1)
    -- Set nClasses()
    self._numbertargets = torch.max(self.targets,1):squeeze()

    self.uttids = torch.LongTensor():range(1,self:usize())
end


function BaseDataloader:_getsamplelengths()
    local samplelengths = torch.LongTensor(self:usize())
    for i=1,self:usize() do
        -- Get from any subclass the number of audio samples this file has
        samplelengths[i] = self:getNumAudioSamples(readfilelabel(self.filelabels[i]))
    end
    self.samplelengths = samplelengths
    return samplelengths
end

-- Called when sampleiterator is used to index the given files
function BaseDataloader:sampletofeat(samplelengths)
    local runningindex = 0
    local numsamples = 0

    -- set size()
    self._nsamples = samplelengths:sum()

    local sampletofeatid,sampletoclassrange = torch.LongTensor(self._nsamples),torch.LongTensor(self._nsamples)
    for i=1,samplelengths:size(1) do
        numsamples = samplelengths[i]
        -- -- Fill in the filelabel for each sample ( assume they are order equally)
        sampletofeatid[{{runningindex + 1, runningindex + numsamples}}]:fill(i)
        -- Fill in the internal range for each sample
        sampletoclassrange[{{runningindex + 1, runningindex + numsamples}}]:range(1,numsamples)
        runningindex = runningindex + numsamples
    end

    return sampletofeatid,sampletoclassrange
end

function BaseDataloader:_readfilename(filename)
    assert(type(filename) == 'string',"Filename is not a string")
    assert(paths.filep(filename) ~= '',"Filename ".. filename .. "does not exist")
    -- The cache for the filepaths
    local filelabels = torch.CharTensor()
    local targets = torch.LongTensor()
    -- use the commandline to obtain the number of lines and maximum width of a line in linux
    local maxPathLength = tonumber(sys.fexecute("awk '{print $1}' " .. filename .. " | wc -L | cut -f1 -d' '")) + 1
    local nlines = tonumber(sys.fexecute("wc -l "..filename .. " | cut -f1 -d' '" ))
    local numbertargets = tonumber(sys.fexecute("head -n 1 " .. filename .." | tr -d -c ' ' | awk '{ print length; }'")) or 1

    filelabels:resize(nlines, maxPathLength):fill(0)
    targets:resize(nlines,numbertargets):fill(-1)
    local filelabel_data = filelabels:data()
    local target_data = targets:data()
    local targetcount = 0

    -- local headerlengths = torch.LongTensor(nlines)
    -- local headerlengths_data = headerlengths:data()

    local linecount = 0
    local audiosamples = 0

    local curline
    for line in io.lines(filename) do
        curline = line:split(' ')
        -- Labels can be multiple ones
        for i=2,#curline do
            target_data[targetcount]=tonumber(curline[i])
            targetcount = targetcount + 1
        end
        -- Get from any subclass the number of audio samples this file has
        -- audiosamples = self:getNumAudioSamples(curline[1])
        -- -- Save the current sample length
        -- headerlengths_data[linecount] = audiosamples
        -- Copy the current feature path into the featpath chartensor
        ffi.copy(filelabel_data,curline[1])
        -- Skip with the offset of the maxPathLength
        filelabel_data = filelabel_data + maxPathLength
        linecount = linecount + 1
    end
    return filelabels,targets,overall_samples
end

function BaseDataloader:getUtterances(uttids, ... )
    local labels = self.filelabels:index(1,uttids)

    local target = self.targets:index(1,uttids)

    return self:loadAudioUtterance(labels,true),target:view(target:nElement()),labels
end

function BaseDataloader:shuffle()
    self._doshuffle = true

end

function BaseDataloader:sampleiterator(batchsize, epochsize, randomize , ...)
    if not self.sampletofeatid and not self.sampletoclassrange then
        self.sampletofeatid,self.sampletoclassrange = self:sampletofeat(self:_getsamplelengths())
    end
    batchsize = batchsize or 16
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:size()

    local min = math.min


    local sampleids
    if self._doshuffle or randomize  then
        sampleids = torch.LongTensor():randperm(self:size())
    else
        sampleids = torch.LongTensor():range(1,self:size())
    end

    self:beforeIter(unpack(dots))

    local stop,bs
    local batch
    -- Buffers
    local featids,classranges = torch.LongTensor(),torch.LongTensor()
    local labels,target = torch.CharTensor(),torch.LongTensor()
    self._cursample = 1
    -- build iterator
    return function()
        if self._cursample > epochsize then
            self:reset()
            self:afterIter(unpack(dots))
            return
        end
        -- epochsize +1 to let the self._cursample >epochsize trigger after that frame.
        bs = min(self._cursample+batchsize, epochsize + 1) - self._cursample

        stop = self._cursample + bs - 1
        local cursampleids = sampleids[{{self._cursample,stop}}]
        -- the row from the file lists
        featids:index(self.sampletofeatid,1,cursampleids)
        -- The range of the current sample within the class
        classranges:index(self.sampletoclassrange,1,cursampleids)

        -- Labels are passed to the getsample to obtain the datavector
        labels:index(self.filelabels,1,featids)
        target:index(self.targets,1,featids)
        -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
        batch = {self:getSample(labels, classranges, unpack(dots)),target:view(target:nElement())}

        self._cursample = self._cursample + bs
        return self._cursample - 1,epochsize, unpack(batch)
    end
end

-- Iterator which returns whole utterances batched
function BaseDataloader:uttiterator(batchsize,epochsize, randomize, ... )
    batchsize = batchsize or 1
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:usize()
    self._curutterance = 1

    local min = math.min

    local inputs, targets , bs, stop

    local uttids
    if self._doshuffle or randomize  then
        uttids = torch.LongTensor():randperm(self:usize())
    else
        uttids = torch.LongTensor():range(1,self:usize())
    end

    self:beforeIter(unpack(dots))
    -- build iterator
    return function()
        if self._curutterance > epochsize then
            self:afterIter(unpack(dots))
            self:reset()
            return
        end

        bs = min(self._curutterance+batchsize, epochsize + 1 ) - self._curutterance

        stop = self._curutterance + bs - 1
        -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
        local batch = {self:getUtterances(uttids[{{self._curutterance,stop}}], unpack(dots))}
        -- -- allows reuse of inputs and targets buffers for next iteration
        -- inputs, targets = batch[1], batch[2]
        self._curutterance = self._curutterance + bs

        return self._curutterance - 1, epochsize, unpack(batch)
    end
end
-- Resets the current dataloader iterator
function BaseDataloader:reset()
   self._curutterance = 1
   self._cursample = 1
end
