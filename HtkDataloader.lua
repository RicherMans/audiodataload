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


-- Loads the given audiofilepath and subs the given tensor to be in range start,stop. Zerotensor is returned if the stop argument is larger than the audiofile
function HtkDataloader:loadAudioSample(audiofilepath,start,stop,...)
    -- The passed start is calculated over the whole dimensions, thus we pass start equally for each iterator
    -- the calculation just reverts the starting point calculation in getsample()
    return _htktorch.loadsample(audiofilepath,(start-1)/self:dim() + 1)
    -- Return just a vector of zeros. This only happenes when called by Sequenceiterator, otherwise this case is nonexistent
    -- TODO: Fix that problem
    -- if stop - start > htkbuf:size(1) then
    --     return torch.zeros(stop-start + 1)
    -- else
    --     return htkbuf
    -- end

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
function HtkDataloader:getSample(labels, classranges, ...)
    _htktorch = _htktorch or require 'torchhtk'
    -- The stepsize
    local framewindow = self:dim()
    -- Use a local copy of input to make it thread safe
    local inputs = torch.Tensor(labels:size(1),framewindow)
    -- Buffer for audiosample
    local sample = nil
    -- Starting frame
    local framestart = 1
    local frameend = -1
    -- Get the current offset for the data
    for i=1,labels:size(1) do
        framestart = (classranges[i] - 1) * ( framewindow ) + 1
        frameend = framestart+framewindow - 1
        sample = self:loadAudioSample(readfilelabel(labels[i]),framestart,frameend,...)
        inputs[i]:copy(sample)
    end
    -- Ready for cleanup
    sample = nil
    return inputs
end

-- Returns the data dimensions
function HtkDataloader:dim()
    return self._dim
end
