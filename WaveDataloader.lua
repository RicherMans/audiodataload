local adl = require 'audiodataload._base'
local WaveDataloader = torch.class('adl.WaveDataloader', 'adl.BaseDataloader', adl)

require 'audio'


local initcheck = argcheck{
    {
        name="path",
        type="string",
        help="Input file which contains the feature paths",
        check=function(filename)
            if paths.filep(filename) then return true else return false end
        end
    },
    {
        name='framesize',
        type='number',
        help="Size for one frame",
        check = function(num)
            if num > 0  then return true end
        end
    },
    {
        name='shift',
        type='number',
        help="Frameshift, defaults to to same length as framesize (thus not being used)",
        check = function(num)
            if num > 0 then return true end
        end,
        -- Set default of being the framesize
        defaulta='framesize'
    },
}

local function calcnumframes(samplessize,framesize,shift)
    return floor((samplessize-framesize)/shift + 1)
end

local max = math.max

function WaveDataloader:getNumAudioSamples(filename)
    local feat = audio.load(filename)
    return max(1,calcnumframes(feat:size(1),self:dim(),self.shift))
end

function WaveDataloader:__init(...)
    local path,framesize,shift,seqlen,padding, maxseqlen = initcheck(...)
    -- Framesize is the actual dimension of a single sample
    self._dim = framesize
    self.shift = shift
    adl.BaseDataloader.__init(self,path)

    -- self.seqlen = seqlen
    self.padding = padding

end


local audiobuf = nil
local audiobufpath = nil
-- Loads the given audiofilepath and subs the given tensor to be in range start,stop. Zerotensor is returned if the stop argument is larger than the audiofile
function WaveDataloader:loadAudioSample(audiofilepath,start,stop,...)
    if not (audiofilepath == audiobufpath) then
        audiobuf = audio.load(audiofilepath)
    end
    audiobufpath = audiofilepath
    -- Return just a vector of zeros. This only happenes when called by Sequenceiterator, otherwise this case is nonexistent
    if stop > audiobuf:size(1) then
        local bufsize = stop-start + 1
        self._buf = self._buf or torch.Tensor()
        self._buf:resize(bufsize):zero()
        return self._buf
    else
        return audiobuf:sub(start,stop)
    end

end

-- Utterance iterator callback
function WaveDataloader:loadAudioUtterance(audiofilepath,wholeutt,...)
    if audiofilepath:dim() == 2 then
        assert(audiofilepath:size(1) == 1,"Only non batch mode for utterances supported!")
    end
    -- We only pass a single audiofilepath to this function
    audiofilepath = readfilelabel(audiofilepath)

    self._audioloaded = self._audioloaded or torch.Tensor()
    wholeutt = wholeutt or ( wholeutt ~= nil or false )
    self._audioloaded = audio.load(audiofilepath)
    local origaudiosize = self._audioloaded:size(1)
    -- The maximum size of fitting utterances so that no sequence will be mixed with nonzeros and zeros
    local modaudiosize = floor(origaudiosize/self:dim()) * self:dim()
    -- We trim the output if seqlen is small
    local targetsize = self:dim()
    -- return the whole utterance, for single batch cases
    if wholeutt then targetsize = modaudiosize end
    self._buf = self._buf or torch.Tensor()
    self._buf:resize(targetsize):zero()

    self._buf:sub(1,targetsize):copy(self._audioloaded:sub(1,targetsize))
    self._buf = self._buf:view(targetsize/self:dim(),self:dim())
    -- end
    return self._buf
end

-- Iterator callback function
function WaveDataloader:getSample(labels,  ids, ...)

    self._input = self._input or torch.Tensor()
    --  batchsize X dimension
    self._input = self._input:resize(labels:size(1),self:dim())
    -- The Final framesize we gonna extract
    local framewindow = self:dim()
    -- Buffer for audiosample
    local wavesample = nil
    -- Starting frame
    local framestart = 1
    -- ending frame, maximum is the full seqence
    local frameend = -1
    -- Get the current offset for the data
    for i=1,labels:size(1) do
        framestart = (self.sampletoclassrange[ids[i]] - 1) * ( self.shift ) + 1
        frameend = framestart+framewindow - 1
        wavesample = self:loadAudioSample(readfilelabel(labels[i]),framestart,frameend,...)
        self._input[i]:copy(wavesample)
    end
    return self._input,self._target
end


-- Returns the size of the dataset
function WaveDataloader:size()
    return self._nsamples
end

-- Returns the data dimensions
function WaveDataloader:dim()
    return self._dim
end
