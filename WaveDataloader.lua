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
    {
        name='seqlen',
        type='number',
        help="Sequence length of an utterance, if this value is > 1 it forces the output tensor to be Seqlen X batchsize X framesize. If this value is 1 the output of uttiterator is nsamples X framesize.",
        default=1,
        -- Only allow nonzero numbers
        check = function(num)
            if num > 0 then return true end
        end
    },
    {
        name='padding',
        help="Left or right padding for the given utterances, left meaning we append zeros at the beginning of a sequence, right means we append at the end",
        type='string',
        default = 'left',
        check =function (padtype)
            if padtype == 'left' or padtype == 'right' then return true else return false end
        end
    },
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
local function readfilelabel(labels,num)
    if num == nil then
        return ffi.string(torch.data(labels))
    else
        return ffi.string(torch.data(labels[num]))
    end
end

function WaveDataloader:getNumAudioSamples(filename)
    local feat = audio.load(filename)
    return math.max(1,calcnumframes(feat:size(1),self:dim(),self.shift))
end

function WaveDataloader:__init(...)
    local path,framesize,shift,seqlen,padding, maxseqlen = initcheck(...)
    -- Framesize is the actual dimension of a single sample
    self._dim = framesize
    self.shift = shift
    adl.BaseDataloader.__init(self,path)

    self.seqlen = seqlen
    self.padding = padding

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

-- function WaveDataloader:_readfilelengths(filename)
--     assert(type(filename) == 'string',"Filename is not a string")
--     assert(paths.filep(filename) ~= '',"Filename ".. filename .. "does not exist")
--     -- The cache for the filepaths
--     local filelabels = torch.CharTensor()
--     local targets = torch.IntTensor()
--     -- use the commandline to obtain the number of lines and maximum width of a line in linux
--     local maxPathLength = tonumber(sys.fexecute("awk '{print $1}' " .. filename .. " | wc -L | cut -f1 -d' '")) + 1
--     local nlines = tonumber(sys.fexecute("wc -l "..filename .. " | cut -f1 -d' '" ))
--     local numbertargets = tonumber(sys.fexecute("head -n 1 " .. filename .." | tr -d -c ' ' | awk '{ print length; }'")) or 1
--
--     filelabels:resize(nlines, maxPathLength):fill(0)
--     targets:resize(nlines,numbertargets):fill(-1)
--     local str_data = filelabels:data()
--     local target_data = targets:data()
--     local count = 0
--
--     local overall_samples = 0
--
--     local headerlengths = torch.IntTensor(nlines)
--     local headerlengths_data = headerlengths:data()
--
--     local linecount = 0
--     local audiotensor = nil
--     local audiosamples = 0
--     local floor = math.floor
--     local max = math.max
--
--     for line in io.lines(filename) do
--         local l = line:split(' ')
--         -- Labels can be multiple ones
--         for i=2,#l do
--             target_data[count]=tonumber(l[i])
--             -- Go to the next item
--             count = count + 1
--         end
--         audiotensor = audio.load(l[1])
--
--         audiosamples = max(1,calcnumframes(audiotensor:size(1),self:dim()*self.seqlen,self.shift))
--         -- Add to the samples, but do not use the last audiochunk ( since it is not the same size as the others)
--         overall_samples = overall_samples + audiosamples
--
--         -- Load the current header
--         headerlengths_data[linecount] = audiosamples
--         -- Copy the current feature path into the featpath chartensor
--         ffi.copy(str_data,l[1])
--         -- Skip with the offset of the maxPathLength
--         str_data = str_data + maxPathLength
--         linecount = linecount + 1
--     end
--     return filelabels,targets,headerlengths,overall_samples
-- end

-- Loads the given audiofilepath and subs the given tensor to be in range start,stop. Zero padding is applied on either the left or the right side of the tensor, making it possible to use MaskZero()
function WaveDataloader:loadAudioSample(audiofilepath,start,stop,...)
    self._audioloaded = torch.Tensor() or self._audioloaded
    self._audioloaded = audio.load(audiofilepath)
    -- This happens if the seqlength is choosen to be very big, thus we need to pad the input with zeros
    if stop - start > self._audioloaded:size(1) then
        local audiosize = self._audioloaded:size(1)
        -- The maximum size of fitting utterances so that no sequence will be mixed with nonzeros and zeros
        audiosize = floor(audiosize/(self:dim())) * self:dim()
        -- Try to fit only the seqlen utterances in a whole in
        audiosize = floor(audiosize/self.seqlen) * self.seqlen
        local bufsize = stop-start + 1
        self._buf = self._buf or torch.Tensor()
        self._buf:resize(bufsize):zero()
        if self.padding == 'left' then
            self._buf:sub(stop-audiosize+1,bufsize):copy(self._audioloaded:sub(1,audiosize))
        else
            self._buf:sub(1,audiosize):copy(self._audioloaded:sub(1,audiosize))
        end
        return self._buf
    else
        return self._audioloaded:sub(start,stop)
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
    modaudiosize = floor(modaudiosize/self.seqlen) * self.seqlen
    -- We trim the output if seqlen is small
    local targetsize = self:dim()*self.seqlen
    -- return the whole utterance, for single batch cases
    if wholeutt then targetsize = modaudiosize end
    self._buf = self._buf or torch.Tensor()
    self._buf:resize(targetsize):zero()
    -- If we have more elements as target than in the utterance, we pad. Otherwise no padding is needed , but we truncate the loaded utterance
    if targetsize > modaudiosize then
        if self.padding == 'left' then
            self._buf:sub(targetsize-modaudiosize + 1,targetsize):copy(self._audioloaded:sub(1,modaudiosize))
        else
            self._buf:sub(1,modaudiosize):copy(self._audioloaded:sub(1,modaudiosize))
        end
    else
        self._buf:sub(1,targetsize):copy(self._audioloaded:sub(1,targetsize))
    end
    return self._buf
end

-- Iterator callback functions
function WaveDataloader:getSample(ids, audioloader, ...)

    if not(  self.sampletofeatid and self.sampletoclassrange ) then

        self.sampletofeatid = torch.LongTensor(self:size())
        self.sampletoclassrange = torch.LongTensor(self:size())
        self:sampletofeat(self.samplelengths,self.sampletofeatid,self.sampletoclassrange)
    end

    self._featids = self._featids or torch.LongTensor()
    self._featids = self.sampletofeatid:index(1,ids)
    local labels = self.filelabels:index(1,self._featids)
    local batchdim = 1
    -- The Final framesize we gonna extract
    local framewindow = self:dim()

    self._input = self._input or torch.Tensor()
    self._target = self._target or torch.Tensor()

    self._target = self._target:resize(labels:size(1))


    if self.seqlen > 1 then
        -- Adding another dimensions at the front
        self._input = self._input:resize(self.seqlen,labels:size(1),self:dim())
        batchdim = 2
        -- We extract on top of the frame also the seqlenth
        framewindow = framewindow * self.seqlen
    else
        -- batchsize X dimension
        self._input = self._input:resize(labels:size(1),self:dim())
    end
    -- The targets are unaffected by any seqlen
    self._target:copy(self.targets:index(1,self._featids))

    local wavesample = nil
    -- Starting fram
    local framestart = 1
    -- ending frame, maximum is the full seqence
    local frameend = -1
    -- Get the current offset for the data
    for i=1,labels:size(1) do
        framestart = (self.sampletoclassrange[ids[i]] - 1) * ( self.shift ) + 1
        frameend = framestart+framewindow - 1
        wavesample = audioloader(self,readfilelabel(labels,i),framestart,frameend,...)
        self._input:narrow(batchdim,i,1):copy(wavesample)
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

    if self.seqlen > 1 then
        -- Adding another dimensions at the front, being SEQLEN X 1 X DIM
        self._input = self._input:resize(self.seqlen,numbatches,self:dim())
        local batchdim = 2
        for i=1,numbatches do
            -- Get the current offset for the data
            filelabels[#filelabels + 1] = readfilelabel(labels,i)
            local wavesample = self:loadAudioUtterance(filelabels[#filelabels])
            self._input:narrow(batchdim,i,1):copy(wavesample)
        end
    -- In case we have only one batch, we return the whole tensor as in size NFRAMES (BATCHDIM) X Targetdim
    else
        filelabels[#filelabels + 1] = readfilelabel(labels,1)
        local wavesample = audioloader(self,filelabels[#filelabels],true)
        self._input:resize(wavesample:size(1)/self:dim(),self:dim()):copy(wavesample)
    end

    return self._input, self._target, filelabels
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
--
-- function WaveDataloader:nClasses()
--     return self._numbertargets
-- end
--
-- -- Returns the size of the utterances
-- function WaveDataloader:usize()
--     return self._numutterances
-- end

-- Returns the data dimensions
function WaveDataloader:dim()
    return self._dim
end
