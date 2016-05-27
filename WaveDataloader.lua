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
    -- {
    --     name='seqlen',
    --     type='number',
    --     help="Sequence length of an utterance, if this value is > 1 it forces the output tensor to be Seqlen X batchsize X framesize. If this value is 1 the output of uttiterator is nsamples X framesize.",
    --     default=1,
    --     -- Only allow nonzero numbers
    --     check = function(num)
    --         if num > 0 then return true end
    --     end
    -- },
    -- {
    --     name='padding',
    --     help="Left or right padding for the given utterances, left meaning we append zeros at the beginning of a sequence, right means we append at the end",
    --     type='string',
    --     default = 'left',
    --     check =function (padtype)
    --         if padtype == 'left' or padtype == 'right' then return true else return false end
    --     end
    -- },
    -- {
    --     name='usemaxseqlength',
    --     type='boolean',
    --     default=false,
    --     help='If this parameter is set, seqlen is ignored and during utterance iteration, the tensor size will be maxseqlen x batchsize x framesize. This function is useful for cases zero-padded batches are needed during evaluation.'
    --
    -- }
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
-- Loads the given audiofilepath and subs the given tensor to be in range start,stop. Zero padding is applied on either the left or the right side of the tensor, making it possible to use MaskZero()
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
        -- local audiosize = audiobuf:size(1)
        -- -- The maximum size of fitting utterances so that no sequence will be mixed with nonzeros and zeros
        -- audiosize = floor(audiosize/(self:dim())) * self:dim()
        -- -- Try to fit only the seqlen utterances in a whole in
        -- -- audiosize = floor(audiosize/self.seqlen) * self.seqlen
        -- self._buf:sub(1,audiosize):copy(audiobuf:sub(1,audiosize))
        -- if self.padding == 'left' then
        --     self._buf:sub(stop-audiosize+1,bufsize):copy(audiobuf:sub(1,audiosize))
        -- else
        -- end
        -- return self._buf
    else
        return audiobuf:sub(start,stop)
    end

end

function WaveDataloader:loadAudioUtterance(audiofilepath,wholeutt,...)
    self._audioloaded = torch.Tensor() or self._audioloaded
    wholeutt = wholeutt or ( wholeutt ~= nil or false )
    self._audioloaded = audio.load(audiofilepath)
    local origaudiosize = self._audioloaded:size(1)
    -- The maximum size of fitting utterances so that no sequence will be mixed with nonzeros and zeros
    local modaudiosize = floor(origaudiosize/self:dim()) * self:dim()
    -- Try to fit only the seqlen utterances in a whole in
    -- modaudiosize = floor(modaudiosize/self.seqlen) * self.seqlen
    -- We trim the output if seqlen is small
    local targetsize = self:dim()
    -- return the whole utterance, for single batch cases
    if wholeutt then targetsize = modaudiosize end
    self._buf = self._buf or torch.Tensor()
    self._buf:resize(targetsize):zero()
    -- If we have more elements as target than in the utterance, we pad. Otherwise no padding is needed , but we truncate the loaded utterance
    -- if targetsize > modaudiosize then
    --     if self.padding == 'left' then
    --         self._buf:sub(targetsize-modaudiosize + 1,targetsize):copy(self._audioloaded:sub(1,modaudiosize))
    --     else
    --         self._buf:sub(1,modaudiosize):copy(self._audioloaded:sub(1,modaudiosize))
    --     end
    -- else
    self._buf:sub(1,targetsize):copy(self._audioloaded:sub(1,targetsize))
    -- end
    return self._buf
end

-- Iterator callback functions
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
        wavesample = self:loadAudioSample(readfilelabel(labels,i),framestart,frameend,...)
        self._input[i]:copy(wavesample)
    end
    return self._input,self._target
end

-- returns a whole utterance, either chunked into batches X dim or if seqlen is specified the data goes into the seqlen , batchdim will be left as one
function WaveDataloader:getUtterance(uttids,audioloader,...)
    local numbatches = uttids:size(1)
    local labels = self.filelabels:index(1,uttids)


    self._input = self._input or torch.Tensor()
    self._target = self._target or torch.Tensor()

    self._target = self._target:resize(numbatches):copy(self.targets:index(1,uttids))

    local filelabels = {}

    -- if self.seqlen > 1 then
    --     -- Adding another dimensions at the front, being SEQLEN X 1 X DIM
    --     self._input = self._input:resize(self.seqlen,numbatches,self:dim())
    --     local batchdim = 2
    --     for i=1,numbatches do
    --         -- Get the current offset for the data
    --         filelabels[#filelabels + 1] = readfilelabel(labels,i)
    --         local wavesample = self:loadAudioUtterance(filelabels[#filelabels])
    --         self._input:narrow(batchdim,i,1):copy(wavesample)
    --     end
    -- In case we have only one batch, we return the whole tensor as in size NFRAMES (BATCHDIM) X Targetdim
    -- else
    filelabels[#filelabels + 1] = readfilelabel(labels,1)
    local wavesample = self:loadAudioUtterance(filelabels[#filelabels],true)
    self._input:resize(wavesample:size(1)/self:dim(),self:dim()):copy(wavesample)
    -- end

    return self._input, filelabels
end


-- Returns the size of the dataset
function WaveDataloader:size()
    return self._nsamples
end

-- Returns the data dimensions
function WaveDataloader:dim()
    return self._dim
end
