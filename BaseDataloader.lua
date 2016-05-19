local adl = require '_base'
local BaseDataloader = torch.class('adl.BaseDataloader', adl)


-- Randomizes the input
function BaseDataloader:random()
    error "Not implemented"
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


function BaseDataloader:subSamples(batchsize,start,stop, ... )
    self._sampleid = self._sampleid or torch.LongTensor()
    self._sampleid:resize(batchsize):randperm(batchsize)
    return self:getSample(self._sampleid,start,stop,...)
end

function BaseDataloader:getSlice(start,stop, ... )
    self._utts = self._utts or torch.LongTensor()
    self._utts:range(start,stop)
    return self:getUtterance(self._utts,start,stop,...)
end

function BaseDataloader:sampleiterator(batchsize,epochsize,randomize,...)
    batchsize = batchsize or 16
    local dots = {...}
    local size = self:size()
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:size()
    local numsamples = 0

    local min = math.min

    local inputs, targets
    randomize = randomize or (randomize == nil or randomize )

    -- build iterator
    return function()
        if numsamples >= epochsize then
            return
        end

        local bs = min(numsamples+batchsize, epochsize) - numsamples
        -- inputs and targets
        local batch = {self:subSamples(bs, inputs, targets, unpack(dots))}
        -- allows reuse of inputs and targets buffers for next iteration
        inputs, targets = batch[1], batch[2]

        numsamples = numsamples + bs

        return numsamples, unpack(batch)
    end
end

function BaseDataloader:uttiterator(utt,epochsize, ... )

end
