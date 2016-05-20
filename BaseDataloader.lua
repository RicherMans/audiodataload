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

function BaseDataloader:getSample(ids,start,stop,...)
    error "Not Implemented"
end


function BaseDataloader:subSamples(start,stop, ... )
    self._sampleid = self._sampleid or torch.LongTensor()
    self._sampleid:resize(stop-start+1):range(start,stop)
    return self:getSample(self._sampleid,...)
end

function BaseDataloader:getUtterances(uttid, ... )
    return self:getUtterance(uttid,...)
end

-- Loads a single audio file into the memory. Used to overload by other classes and should be called during getSample()
function BaseDataloader:loadAudioSample(audiofilepath,start,stop,...)
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
    -- build iterator
    return function()
        if numsamples >= epochsize then
            return
        end

        local bs = min(numsamples+batchsize, epochsize) - numsamples

        local stop = numsamples + bs
        -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
        local batch = {self:subSamples(numsamples, stop, unpack(dots))}
        -- -- allows reuse of inputs and targets buffers for next iteration
        -- inputs, targets = batch[1], batch[2]

        numsamples = numsamples + bs
        return numsamples,epochsize, unpack(batch)
    end
end

-- Iterator which returns whole utterances batched
function BaseDataloader:uttiterator(batchsize,epochsize, ... )
    batchsize = batchsize or 16
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:usize()
    local numsamples = 1

    local min = math.min

    local inputs, targets , bs, stop
    -- build iterator
    return function()
        if numsamples >= epochsize then
            return
        end

        bs = min(numsamples+batchsize, epochsize) - numsamples

        stop = numsamples + bs
        -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
        local batch = {self:getUtterances(numsamples, stop, unpack(dots))}
        -- -- allows reuse of inputs and targets buffers for next iteration
        -- inputs, targets = batch[1], batch[2]

        numsamples = numsamples + bs

        return numsamples,epochsize, unpack(batch)
    end
end
