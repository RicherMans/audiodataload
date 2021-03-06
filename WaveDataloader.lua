local adl = require 'audiodataload._base'
local WaveDataloader = torch.class('adl.WaveDataloader', 'adl.BaseDataloader', adl)


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

-- Loads a wavefile and rescales the output ( there is a small bug with libsox that it scales all values by 2^16)
local function loadwave(filepath)
    return audio.load(filepath):div(2^16)
end

function WaveDataloader:getNumAudioSamples(filename)
    local feat = loadwave(filename)
    local nframes =  calcnumframes(feat:size(1),self:dim(),self.shift)
    if (nframes < 1) then nframes = 1 end
    return nframes
end

function WaveDataloader:__init(...)
    require 'audio'
    local path,framesize,shift = initcheck(...)
    -- Framesize is the actual dimension of a single sample
    self._dim = framesize
    self.shift = shift
    adl.BaseDataloader.__init(self,path)

end


-- Loads the given audiofilepath and subs the given tensor to be in range start,stop. Zerotensor is returned if the stop argument is larger than the audiofile
function WaveDataloader:loadAudioSample(audiofilepath,start,stop,...)
    local audiobuf = loadwave(audiofilepath)
    -- -- Return just a vector of zeros. This only happenes when called by Sequenceiterator, otherwise this case is nonexistent
    -- if stop > audiobuf:size(1) then
    --     return torch.zeros(stop-start + 1)
    -- else
    return audiobuf:sub(start,stop)
    -- end

end

-- Utterance iterator callback
function WaveDataloader:loadAudioUtterance(audiofilepath,wholeutt,...)
    if audiofilepath:dim() == 2 then
        assert(audiofilepath:size(1) == 1,"Only non batch mode for utterances supported!")
    end
    -- We only pass a single audiofilepath to this function
    audiofilepath = readfilelabel(audiofilepath)

    wholeutt = wholeutt or ( wholeutt ~= nil or false )
    local audioloaded = loadwave(audiofilepath)
    -- The maximum size of fitting utterances so that no sequence will be mixed with nonzeros and zeros
    local modaudiosize = floor(audioloaded:size(1)/self:dim()) * self:dim()
    -- We trim the output if seqlen is small
    local targetsize = self:dim()
    local numtsteps = calcnumframes(modaudiosize,self:dim(),self.shift)
    local outputsize = numtsteps*self:dim()
    -- return the whole utterance, for single batch cases
    if wholeutt then targetsize = outputsize end
    
    local buf = torch.zeros(outputsize)
    local startbuf,startaud = 1,1
    local stopbuf,stopaud = self:dim(),self:dim()

    for i=1,numtsteps do
        buf:sub(startbuf,stopbuf):copy(audioloaded:sub(startaud,stopaud))
        startbuf = startbuf + self:dim() 
        stopbuf = stopbuf + self:dim()
        startaud = i * self.shift + 1
        stopaud = startaud + self:dim() - 1
    end
    -- end
    return buf:view(outputsize/self:dim(),self:dim())
end

-- Iterator callback function
function WaveDataloader:getSample(labels,  classrange, ...)
    --  batchsize X dimension
    local input = torch.Tensor(labels:size(1),self:dim())
    -- Buffer for audiosample
    local wavesample = nil

    local startsample = 0
    local endsample = 0
    -- Get the current offset for the data
    for i=1,labels:size(1) do
        startsample = (classrange[i] - 1) * ( self.shift ) + 1
        endsample = startsample + self:dim() - 1
        wavesample = self:loadAudioSample(readfilelabel(labels[i]),startsample,endsample,...)
        input[i]:copy(wavesample)
    end
    return input
end


-- Returns the data dimensions
function WaveDataloader:dim()
    return self._dim
end
