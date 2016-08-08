local adl = require 'audiodataload._base'
local HtkCacheiterator = torch.class('adl.HtkCacheiterator', 'adl.BaseDataloader',adl)

local initcheck = argcheck{
    pack=true,
    {
        name='module',
        type='adl.BaseDataloader',
        help="The module to wrap around",
    },
    {
        name='dirpath',
        type='string',
        help="The directory where the parts are dumped to",
    },
    {
        name='cachesize',
        type='number',
        help="If newfile is provided, this arguments specifies the average chunksize for a deflated file, thus also enabling deflation.",
        default=1000,
    }

}


local function dump(cache,outputfile)
    local tensor = torch.concat(cache,1)
    _htktorch.write(outputfile,tensor,"USER")
end

function HtkCacheiterator:__init(...)
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end
    for k,v in pairs(self.module) do self[k] = v end

    _htktorch = _htktorch or require 'torchhtk'

    local size = self.cachesize

    local function include(kwd)
        return function(fname) 
            if fname:find(kwd) then return true end
        end
    end

    if not paths.dirp(self.dirpath) then
        paths.mkdir(self.dirpath)
        local filetopos = {}
        local filetodump = {}
        local runningid = 1
        local outputfileid = 1
        local outputfile
        local cache = {}
        for done,finished,input,_,path in self.module:uttiterator() do
            outputfile=paths.concat(self.dirpath,"part_"..outputfileid)
            path = readfilelabel(path)
            filetopos[path] = runningid
            filetodump[path] = outputfile
            runningid = runningid + input:size(1)
            cache[#cache+1] = input
            if runningid > size then
                outputfileid = outputfileid + 1
                runningid = 1
                dump(cache,outputfile)
                cache=nil
                cache={}
            end
        end
        -- Dump the rest of the 
        if cache then
            dump(cache,outputfile)
        end
        self.filetopos = filetopos
        self.filetodump = filetodump
    end
end

function HtkCacheiterator:loadAudioSample(audiofilepath,start,stop,...)
    -- print(self.filetopos[audiofilepath],audiofilepath,start,stop,self.filetopos[audiofilepath]+((start-1)/self:dim()))
    local offset = self.filetopos[audiofilepath]+((start-1)/self:dim() ) 
    local audiopath = self.filetodump[audiofilepath]
    return _htktorch.loadsample(audiopath,offset)
end

-- Samples are not extra handled, just use the wrapped classes
function HtkCacheiterator:getSample(labels,  classranges, ...)
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


function HtkCacheiterator:loadAudioUtterance(audiofilepath,wholeutt)
    error("Not implemented")
end



-- Number of utterances in the dataset
-- Just wrap it around the moduel
function HtkCacheiterator:usize()
    return self.module:usize()
end

-- Returns the sample size of the dataset
function HtkCacheiterator:size()
    return self.module:size()
end

-- Returns the data dimensions
function HtkCacheiterator:dim()
    return self.module:dim()
end

function HtkCacheiterator:nClasses()
    return self.module:nClasses()
end