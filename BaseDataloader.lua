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

    local filelabels,targets,samplelengths,overall_samples = self:_readfilename(filename)

    -- Set the filelabels and targets for the subclasses
    self.filelabels = filelabels
    self.targets = targets
    self.samplelengths = samplelengths

    -- set usize()
    self._uttsize = self.filelabels:size(1)
    -- set size()
    self._nsamples = overall_samples
    -- Set nClasses()
    self._numbertargets = torch.max(self.targets,1):squeeze()

    self.sampleids = torch.LongTensor():range(1,self:size())
    self.uttids = torch.LongTensor():range(1,self:usize())
end


function BaseDataloader:sampletofeat(samplelengths)
    local runningindex = 0
    local numsamples = 0
    local sampletofeatid,sampletoclassrange = torch.LongTensor(self:size()),torch.LongTensor(self:size())
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
    local str_data = filelabels:data()
    local target_data = targets:data()
    local targetcount = 0

    local overall_samples = 0

    local headerlengths = torch.LongTensor(nlines)
    local headerlengths_data = headerlengths:data()

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
        audiosamples = self:getNumAudioSamples(curline[1])
        -- Add to the samples, but do not use the last audiochunk ( since it is not the same size as the others)
        overall_samples = overall_samples + audiosamples

        -- Load the current header
        headerlengths_data[linecount] = audiosamples
        -- Copy the current feature path into the featpath chartensor
        ffi.copy(str_data,curline[1])
        -- Skip with the offset of the maxPathLength
        str_data = str_data + maxPathLength
        linecount = linecount + 1
    end
    return filelabels,targets,headerlengths,overall_samples
end

function BaseDataloader:getUtterances(uttids, ... )
    local labels = self.filelabels:index(1,uttids)

    local target = self.targets:index(1,uttids)

    return self:loadAudioUtterance(labels,true),target:view(target:nElement()),labels
end

function BaseDataloader:shuffle()
    self.sampleids = torch.LongTensor():randperm(self:size())
    self.uttids = torch.LongTensor():randperm(self:usize())
end

function BaseDataloader:sampleiterator(batchsize, epochsize, ...)
    batchsize = batchsize or 16
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:size()

    if not ( self.sampletofeatid and self.sampletoclassrange ) then
        self.sampletofeatid,self.sampletoclassrange = self:sampletofeat(self.samplelengths,self.targets)
    end

    local min = math.min

    self:beforeIter(unpack(dots))

    local stop,bs
    local batch
    local featids,classranges = torch.LongTensor(),torch.LongTensor()
    local labels,target = torch.CharTensor(),torch.LongTensor()
    self._cursample = 1
    -- build iterator
    return function()
        if self._cursample > epochsize then
            self:afterIter(unpack(dots))
            return
        end
        -- epochsize +1 to let the self._cursample >epochsize trigger after that frame. 
        bs = min(self._cursample+batchsize, epochsize + 1) - self._cursample

        stop = self._cursample + bs - 1
        local cursampleids = self.sampleids[{{self._cursample,stop}}]
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
function BaseDataloader:uttiterator(batchsize,epochsize, ... )
    batchsize = batchsize or 1
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:usize()
    self._curutterance = 1

    local min = math.min

    local inputs, targets , bs, stop

    self:beforeIter(unpack(dots))
    -- build iterator
    return function()
        if self._curutterance > epochsize then
            self:afterIter(unpack(dots))
            return
        end

        bs = min(self._curutterance+batchsize, epochsize + 1 ) - self._curutterance

        stop = self._curutterance + bs - 1
        -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
        local batch = {self:getUtterances(self.uttids[{{self._curutterance,stop}}], unpack(dots))}
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
