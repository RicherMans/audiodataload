local adl = require '_base'
local WaveDataloader = torch.class('adl.WaveDataloader', 'adl.BaseDataloader', adl)

local ffi = require 'ffi'
local argcheck = require 'argcheck'
require 'audio'


initcheck = argcheck{
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
    }
}

function WaveDataloader:__init(...)
    local path,framesize = initcheck(...)
    print(path,framesize)

    -- self:_prepare(args.path,args.framesize)
end

function WaveDataloader:_prepare(filename,framesize)
    assert(filename ~= '',"Filename is empty! ")
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
    local audiosamples = nil
    local floor = math.floor

    for line in io.lines(filename) do
        local l = line:split(' ')
        -- Labels can be multiple ones
        for i=2,#l do
            target_data[count]=tonumber(l[i])
            -- Go to the next item
            count = count + 1
        end
        audiotensor = audio.load(l[1])
        -- The number of audiosamples for the current framesize
        -- If we have zero length, we just pad, but assum ehaving a single utterance
        audiosamples = max(1,floor(audiotensor:size(1)/framesize))
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

function WaveDataloader:getSample(ids,start,stop)
end

-- Returns the size of the dataset
function WaveDataloader:size()
    return self._nsamples
end

-- Returns the data dimensions
function WaveDataloader:dim()
    return self._dim
end
