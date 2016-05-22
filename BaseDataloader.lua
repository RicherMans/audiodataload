local adl = require '_base'
local BaseDataloader = torch.class('adl.BaseDataloader', adl)


-- Randomizes the input
function BaseDataloader:random()
    error "Not implemented"
end

-- Number of utterances in the dataset
function BaseDataloader:usize()
    error "Not Implemented"
end

-- Returns the sample size of the dataset
function BaseDataloader:size()
    error "Not implemented"
end

-- Returns the data dimensions
function BaseDataloader:dim()
    error "Not Implemented"
end

function BaseDataloader:getSamples(ids,start,stop,...)
    local labels = self:prepareSample()
    local start,stop
    for i=1,labels:size(1) do
        start,stop = self:getStartEnd()
        self:loadSample(labels[i],start,stop,...)
    end
    return self:returnSample()
end

function BaseDataloader:prepareSample()
    error "Not Implemented"
end

function BaseDataloader:getStartEnd()
    error "Not Implemented"
end

function BaseDataloader:loadSample(label,start,stop,...)
    error "Not Implemented"
end

-- These two methods should be called within the getSample/getutterance methods
-- to load a single sample/utterance out
-- function BaseDataloader:loadAudioSample(audiofilepath,...)
--     error "Not Implemented"
-- end
--
-- function BaseDataloader:loadAudioUtterance(audiofilepath,...)
--     error "Not Implemented"
-- end

-- templates to callback the iterators
function BaseDataloader:beforeIter(...)
    return
end
-- Callback for any tasks after finishing the iteration e.g. closing files
function BaseDataloader:afterIter(...)
    return
end

function BaseDataloader:subSamples(start,stop, audioloader, ... )
    self._sampleid = self._sampleid or torch.LongTensor()
    self._sampleid:resize(stop-start + 1):range(start,stop)
    return self:getSample(self._sampleid, audioloader, ...)
end

function BaseDataloader:getUtterances(start,stop,audioloader, ... )
    self._utteranceids = self._utteranceids or torch.LongTensor()
    self._utteranceids:resize(stop-start+1):range(start,stop)
    return self:getUtterance(self._utteranceids,audioloader,...)
end

-- Loads a single audio file into the memory. Used to overload by other classes and should be called during getSample()
function BaseDataloader:loadAudioSample(audiofilepath,start,stop,...)
    error "Not Implemented"
end

function BaseDataloader:getAudioLoader()
    error "Not Implemented"
end

function BaseDataloader:sampleiterator(batchsize,epochsize,...)
    batchsize = batchsize or 16
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:size()
    local numsamples = 1

    local min = math.min

    local inputs, targets

    self:beforeIter(unpack(dots))
    local audioloader = self.loadAudioSample
    -- build iterator
    return function()
        if numsamples > epochsize then
            self:afterIter(unpack(dots))
            return
        end

        local bs = min(numsamples+batchsize, epochsize + 1) - numsamples


        local stop = numsamples + bs - 1
        -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
        local batch = {self:subSamples(numsamples, stop, audioloader, unpack(dots))}
        -- -- allows reuse of inputs and targets buffers for next iteration
        -- inputs, targets = batch[1], batch[2]

        numsamples = numsamples + bs
        return numsamples,epochsize, unpack(batch)
    end
end

-- Iterator which returns whole utterances batched
function BaseDataloader:uttiterator(batchsize,epochsize, ... )
    batchsize = batchsize or 1
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:usize()
    local curutterance = 1

    local min = math.min

    local inputs, targets , bs, stop


    self:beforeIter(unpack(dots))
    local audioloader = self.loadAudioUtterance
    -- build iterator
    return function()
        if curutterance > epochsize then
            self:afterIter(unpack(dots))
            return
        end

        bs = min(curutterance+batchsize, epochsize + 1 ) - curutterance

        stop = curutterance + bs - 1
        -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
        local batch = {self:getUtterances(curutterance, stop,audioloader, unpack(dots))}
        -- -- allows reuse of inputs and targets buffers for next iteration
        -- inputs, targets = batch[1], batch[2]

        curutterance = curutterance + bs

        return curutterance, epochsize, unpack(batch)
    end
end
