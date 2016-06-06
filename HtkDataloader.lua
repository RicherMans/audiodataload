local adl = require 'audiodataload._base'
local HtkDataloader = torch.class('adl.HtkDataloader', 'adl.BaseDataloader', adl)


local initcheck = argcheck{
    {
        name="path",
        type="string",
        help="Input file which contains the feature paths",
        check=function(filename)
            if paths.filep(filename) then return true else return false end
        end
    },
}

local max = math.max

-- used by basedataloader to estimate _nsamples
function HtkDataloader:getNumAudioSamples(filename)
    print(_htktorch.loadheader(filename).nsamples)
    return _htktorch.loadheader(filename).nsamples
end


function HtkDataloader:__init(...)
    local path = initcheck(...)
    _htktorch = _htktorch or require 'torchhtk'
    -- Framesize is the actual dimension of a single sample
    adl.BaseDataloader.__init(self,path)
    -- Get a sample from the first item, and use it to obtain the data dimension
    local firstfile = readfilelabel(self.filelabels[1])
    local header = _htktorch.loadheader(firstfile)

    self._dim = header.samplesize

end


local htkbuf = nil
local audiobufpath = nil
-- Loads the given audiofilepath and subs the given tensor to be in range start,stop. Zerotensor is returned if the stop argument is larger than the audiofile
function HtkDataloader:loadAudioSample(audiofilepath,start,stop,...)
    if not (audiofilepath == audiobufpath) then
        htkbuf = _htktorch.load(audiofilepath)
        htkbuf = htkbuf:view(htkbuf:nElement())
    end
    audiobufpath = audiofilepath
    -- Return just a vector of zeros. This only happenes when called by Sequenceiterator, otherwise this case is nonexistent
    if stop > htkbuf:size(1) then
        return torch.zeros(stop-start + 1)
    else
        return htkbuf:sub(start,stop)
    end

end

-- Utterance iterator callback
function HtkDataloader:loadAudioUtterance(audiofilepath,wholeutt,...)
    if audiofilepath:dim() == 2 then
        assert(audiofilepath:size(1) == 1,"Only non batch mode for utterances supported!")
    end
    -- We only pass a single audiofilepath to this function
    audiofilepath = readfilelabel(audiofilepath)

    return _htktorch.load(audiofilepath)
end

-- Iterator callback function
function HtkDataloader:getSample(labels,  ids, ...)

    self._input = self._input or torch.Tensor()
    --  batchsize X dimension
    self._input = self._input:resize(labels:size(1),self:dim())
    -- The stepsize
    local framewindow = self:dim()
    -- Buffer for audiosample
    local sample = nil
    -- Starting frame
    local framestart = 1
    -- ending frame, maximum is the full seqence
    local frameend = -1
    -- Get the current offset for the data
    for i=1,labels:size(1) do
        framestart = self.sampletoclassrange[ids[i]]
        frameend = framestart+framewindow - 1
        sample = self:loadAudioSample(readfilelabel(labels[i]),framestart,frameend,...)
        self._input[i]:copy(sample)
    end
    return self._input,self._target
end

-- Returns the data dimensions
function HtkDataloader:dim()
    return self._dim
end
