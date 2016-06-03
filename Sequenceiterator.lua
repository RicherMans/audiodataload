local adl = require 'audiodataload._base'
local Sequenceiterator = torch.class('adl.Sequenceiterator', 'adl.BaseDataloader', adl)

local mathmax = math.max
local mathceil = math.ceil

local initcheck = argcheck{
    pack=true,
    {
        name='wrappedmodule',
        type='adl.BaseDataloader',
        help="The module to wrap around",
    },
    {
        name='seqlen',
        type='number',
        help="Sequence length of an utterance, if this value is > 1 it forces the output tensor to be Seqlen X batchsize X framesize.",
        default=1,
        -- Only allow nonzero numbers
        check = function(num)
            if num > 0 then return true end
        end
    },
    {
        name='usemaxseqlength',
        type='boolean',
        default=false,
        help='If this parameter is set, seqlen is ignored and during utterance iteration, the tensor size will be maxseqlen x batchsize x framesize. This function is useful for cases zero-padded batches are needed during evaluation.'
    },
    {
        name='padding',
        help="Left or right padding for the given utterances, left meaning we append zeros at the beginning of a sequence, right means we append at the end",
        type='string',
        default = 'left',
        check = function (padtype)
            if padtype == 'left' or padtype == 'right' then return true else return false end
        end
    },

}

function Sequenceiterator:__init(...)
    local args = initcheck(...)
    for k,v in pairs(args) do
        self[k] =v
    end
    -- Copy the wrapped modules members
    for k,v in pairs(self.wrappedmodule) do
        self[k]=v
    end

    -- Overwrite seqlength
    if self.usemaxseqlength then
        local maxseqlen = self.wrappedmodule.samplelengths:max()
        if maxseqlen == 0 then
            error "Given framesize is too large for the dataset"
        end
        self.seqlen = maxseqlen
    end

    -- same size as samplelengths
    local onetensor = self.wrappedmodule.samplelengths:clone():fill(1)
    -- Do for each utterance the size calculation. When we sequence any utterance, it can be said that we just remove the last self.seqlen frames from this utterance
    local numsamples = torch.add(self.wrappedmodule.samplelengths,-self.seqlen)
    -- we have always at least 1 one seqence for each utterance
    numsamples = torch.cmax(onetensor,numsamples)

    self._nsamples = numsamples:sum()

end



-- Iterator callback functions
function Sequenceiterator:getSample(labels,  ids, ...)

    self._input = self._input or torch.Tensor()
    --  batchsize X dimension
    self._input = self._input:resize(self.seqlen,labels:size(1),self:dim())
    -- The Final framesize we gonna extract
    local framewindow = self:dim()
    -- Buffer for audiosample
    local sample
    -- Starting frame
    local framestart = 1
    -- ending frame, maximum is the full seqence
    local frameend = -1
    -- Get the current datalabel
    local curlabel
    -- print(self.sampletoclassrange)
    for i=1,labels:size(1) do
        -- Parameter shift is hijakcked from the wrapped module
        framestart = (self.sampletoclassrange[ids[i]] - 1) * ( self.shift ) + 1
        frameend = framestart+framewindow - 1
        curlabel = readfilelabel(labels[i])
        local tic = torch.tic()
        for j=1,self.seqlen do
            sample = self.wrappedmodule:loadAudioSample(curlabel,framestart,frameend,...)
            -- Shift the window to the next
            framestart = framestart + framewindow
            frameend = frameend + framewindow
            self._input[{{j},{i}}]:copy(sample)
        end
    end
    return self._input,self._target
end


function Sequenceiterator:loadAudioUtterance(audiofilepaths,...)
    local audiofeat
    -- Final size of the output utterance
    local targetsize = self:dim() * self.seqlen
    local framewindow = self:dim()
    local buf = torch.Tensor(self.seqlen,audiofilepaths:size(1),self:dim())
    for i=1,audiofilepaths:size(1) do
        audiofeat = self.wrappedmodule:loadAudioUtterance(audiofilepaths[i],true)
        local origaudiosize = audiofeat:nElement()

        -- The maximum size of fitting utterances so that no sequence will be mixed with nonzeros and zeros
        local modaudiosize = floor(origaudiosize/framewindow) * framewindow
        -- Try to fit only the seqlen utterances in a whole in
        modaudiosize = floor(modaudiosize/self.seqlen) * self.seqlen
        -- If we have more elements as target than in the utterance, we pad. Otherwise no padding is needed , but we truncate the loaded utterance
        if targetsize > modaudiosize then
            local seqtargetsize = floor(targetsize / framewindow)
            local seqmodaudiosize = floor(modaudiosize / framewindow)
            if self.padding == 'left' then
                buf[{{seqtargetsize-seqmodaudiosize + 1,seqtargetsize},{i}}]:copy(audiofeat:view(origaudiosize):sub(1,seqmodaudiosize* framewindow))
            else
                buf[{{1,seqmodaudiosize},{i}}]:copy(audiofeat:view(origaudiosize):sub(1,seqmodaudiosize*framewindow))
            end
        else
            buf[{{},{i}}]:copy(audiofeat:view(origaudiosize):sub(1,targetsize))
        end
    end

    return buf
end


function Sequenceiterator:sampletofeat(samplelengths)
    local runningindex = 0
    local numsamples = 0
    local sampletofeatid,sampletoclassrange = torch.LongTensor(self:size()),torch.LongTensor(self:size())
    local acc = 0
    for i=1,samplelengths:size(1) do
        -- use at least one sample ( zero pad it later)
        numsamples = mathmax(1,samplelengths[i] - self.seqlen)
        acc = acc + numsamples
        -- -- Fill in the target for each sample
        sampletofeatid[{{runningindex + 1, runningindex + numsamples}}]:fill(i)
        -- Fill in the internal range for each sample
        sampletoclassrange[{{runningindex + 1, runningindex + numsamples}}]:range(1,numsamples)

        runningindex = runningindex + numsamples
    end

    return sampletofeatid, sampletoclassrange
end


-- Number of utterances in the dataset
-- Just wrap it around the moduel
function Sequenceiterator:usize()
    return self.wrappedmodule:usize()
end

-- Returns the sample size of the dataset
function Sequenceiterator:size()
    return self._nsamples
end

-- Returns the data dimensions
function Sequenceiterator:dim()
    return self.wrappedmodule:dim()
end

function Sequenceiterator:nClasses()
    return self.wrappedmodule:nClasses()
end
