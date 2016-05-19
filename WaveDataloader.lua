local adl = require '_base'
local WaveDataloader = torch.class('adl.WaveDataloader', 'adl.BaseDataloader', adl)

local ffi = require 'ffi'
local argcheck = require 'argcheck'
require 'audio'


local initcheck = argcheck{
    quiet=true,
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
    },
    {
        name='shift',
        type='number',
        help="Frameshift, defaults to to same length as framesize (thus not being used)",
        -- Set default of being the framesize
        defaulta='framesize'
    },
    {
        name='seqlen',
        type='number',
        help="Sequence length of an utterance, if this value is > 1 it forces the output tensor to be Seqlen X batchsize X framesize",
        default=1
    },
    {
        name='padding',
        help="Left or right padding for the given utterances, if we need any",
        type='string',
        check =function (padtype)
            if padtype == 'left' or padtype == 'right' then return true else return false end
        end
    }
}

local function calcnumframes(samplessize,framesize,shift)
    return math.floor((samplessize-framesize)/shift + 1)
end
local function readfilelabel(labels,num)
    return ffi.string(torch.data(labels[num]))
end

function WaveDataloader:__init(...)
    local this,path,framesize,shift,seqlen,padding = initcheck(...)

    -- Framesize is the actual dimension of a single sample
    self._dim = framesize
    self.shift = shift
    self.seqlen = seqlen

    local filelabels,targets,headerlengths, overall_samples = self:_readfilelengths(path)

    self.filelabels = filelabels
    self.targets = targets
    self._nsamples = overall_samples

    self.sampletofeatid, self.sampletoclassrange = self:_headerstosamples(headerlengths,self:size())


end

--
function WaveDataloader:_headerstosamples(samplelengths,overall_samples)
    local sampletofeatid = torch.LongTensor(overall_samples)
    local sampletoclassrange = torch.LongTensor(overall_samples)
    local runningindex = 0
    local numsamples = 0
    for i=1,samplelengths:size(1) do
        numsamples  = samplelengths[i]
        -- -- Fill in the target for each sample
        sampletofeatid[{{runningindex + 1, runningindex + numsamples}}]:fill(i)
        sampletoclassrange[{{runningindex + 1, runningindex + numsamples}}]:range(1,numsamples)
        runningindex = runningindex + numsamples
    end

    return sampletofeatid,sampletoclassrange
end

function WaveDataloader:_readfilelengths(filename)
    assert(paths.filep(filename) ~= '',"Filename ".. filename .. "does not exist")
    -- The cache for the filepaths
    local filelabels = torch.CharTensor()
    local targets = torch.IntTensor()
    -- use the commandline to obtain the number of lines and maximum width of a line in linux
    local maxPathLength = tonumber(sys.fexecute("awk '{print $1}' " .. filename .. " | wc -L | cut -f1 -d' '")) + 1
    local nlines = tonumber(sys.fexecute("wc -l "..filename .. " | cut -f1 -d' '" ))
    local numbertargets = tonumber(sys.fexecute("head -n 1 " .. filename .." | tr -d -c ' ' | awk '{ print length; }'")) or 1

    filelabels:resize(nlines, maxPathLength):fill(0)
    targets:resize(nlines,numbertargets):fill(-1)
    local str_data = filelabels:data()
    local target_data = targets:data()
    local count = 0

    local overall_samples = 0
    local runningindex = 0

    local headerlengths = torch.IntTensor(nlines)
    local headerlengths_data = headerlengths:data()

    local linecount = 0
    local audiotensor = nil
    local audiosamples = 0
    local floor = math.floor
    local max = math.max

    for line in io.lines(filename) do
        local l = line:split(' ')
        -- Labels can be multiple ones
        for i=2,#l do
            target_data[count]=tonumber(l[i])
            -- Go to the next item
            count = count + 1
        end
        audiotensor = audio.load(l[1])

        audiosamples = max(1,calcnumframes(audiotensor:size(1),self:dim()*self.seqlen,self.shift))
        -- Add to the samples, but do not use the last audiochunk ( since it is not the same size as the others)
        overall_samples = overall_samples + audiosamples

        -- Load the current header
        headerlengths_data[linecount] = audiosamples
        -- Copy the current feature path into the featpath chartensor
        ffi.copy(str_data,l[1])
        -- Skip with the offset of the maxPathLength
        str_data = str_data + maxPathLength
        linecount = linecount + 1
    end
    return filelabels,targets,headerlengths,overall_samples
end

function WaveDataloader:getSample(ids)
    self._featids = self._featids or torch.LongTensor()
    self._featids = self.sampletofeatid:index(1,ids)
    local labels = self.filelabels:index(1,self._featids)
    local batchdim = 1
    -- The Final framesize we gonna extract
    local framewindow = self:dim()

    self._input = self._input or torch.Tensor()
    self._target = self._target or torch.Tensor()

    self._input:resize(labels:size(1),self:dim())
    self._target:resize(labels:size(1))

    self._target:copy(self.targets:index(1,self._featids))

    if self.seqlen > 1 then
        -- Adding another dimensions at the front
        self._input = self._input:resize(self.seqlen,-1)
        batchdim = 2
        -- We extract on top of the frame also the seqlenth
        framewindow = framewindow * self.seqlen
    end

    local wavesample = nil
    local batchdimlength = self._input:size(batchdim)
    local framestart = 1
    -- Get the current offset for the data
    for i=1,labels:size(1) do
        framestart = (self.sampletoclassrange[ids[i]] - 1) * ( self.shift ) + 1
        wavesample = audio.load(readfilelabel(labels,i))
        self._input:narrow(batchdim,i,1):copy(wavesample:sub(framestart,framestart+framewindow - 1))
    end
    return self._input,self._target
end

-- randomizes the input sequence
function WaveDataloader:random()
    local randomids = torch.LongTensor():randperm(self:size())
    self.sampletofeatid = self.sampletofeatid:index(1,randomids)
    self.sampletoclassrange = self.sampletoclassrange:index(1,randomids)
end

-- Returns the size of the dataset
function WaveDataloader:size()
    return self._nsamples
end

-- Returns the data dimensions
function WaveDataloader:dim()
    return self._dim
end
