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


local function dump(cache,targets,outputfile)
    local datatensor = torch.concat(cache,1)
    local targettensor = torch.concat(targets,1)
    local tensor = {
        data = datatensor,
        target = targettensor
    }
    torch.save(outputfile,tensor)
end

function HtkCacheiterator:__init(...)
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end
    for k,v in pairs(self.module) do self[k] = v end

    _htktorch = _htktorch or require 'torchhtk'

    local size = self.cachesize
    self.doshuffle = false

    local function include(kwd)
        return function(fname)
            if fname:find(kwd) then return true end
        end
    end

    if not paths.dirp(self.dirpath) then
        paths.mkdir(self.dirpath)
        local dumps = {}
        local runningid = 1
        local outputfileid = 1
        local outputfile
        local datacache = {}
        local targetcache = {}
        for done,finished,input,target,path in self.module:uttiterator() do
            outputfile=paths.concat(self.dirpath,"dump_part_"..outputfileid..".th")
            path = readfilelabel(path)
            runningid = runningid + input:size(1)
            datacache[#datacache+1] = input
            -- Adjust the sizes of the input and the target, since target is only a 1 dim tensor
            targetcache[#targetcache+ 1 ] = target:expand(input:size(1))
            if runningid > size then
                dumps[#dumps+1] = outputfile
                outputfileid = outputfileid + 1
                runningid = 1
                dump(datacache,targetcache,outputfile)
                targetcache = nil
                targetcache = {}
                datacache=nil
                datacache={}
            end
        end
        -- Dump the rest of the data
        if next(datacache) ~= nil then
            dump(datacache,targetcache,outputfile)
            dumps[#dumps+1] = outputfile
        end

        self.dumps = dumps
    end
end

local function batchfeat(feats,targets,size)
    return
        feats:split(size,1),
        targets:split(size,1)
end

function HtkCacheiterator:shuffle()
    self.doshuffle = true
end



function HtkCacheiterator:sampleiterator(batchsize,epochsize,...)
    batchsize = batchsize or 16
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:size()

    local min = math.min

    if not self.sampletofeatid and not self.sampletoclassrange then
         self.sampletofeatid,self.sampletoclassrange = self:sampletofeat(self.samplelengths)
    end

    local wrap = coroutine.wrap
    local yield = coroutine.yield

    self:beforeIter(unpack(dots))
    -- Buffers
    local fname,size = nil,0
    local input,target = torch.Tensor(),torch.LongTensor()

    local function loaddataiter()
        for k,fname in ipairs(self.dumps) do
            if fname == nil then return end
            local data = torch.load(fname)
            -- apply shuffling ( if :shuffle() is called)
            input = data.data
            target = data.target
            size = target:size(1)
            if self.doshuffle then
                local randperm = torch.LongTensor():randperm(size)
                input = input:index(1,randperm)
                target= target:index(1,randperm)
            end

            local inputbatches,targetbatches = batchfeat(input,target,batchsize)
            yield(size,inputbatches,targetbatches)
        end

    end
    local function iterator()
        for size,inpbatches,tgtbatches in wrap(loaddataiter) do
            local done = 1
            for i=1,#inpbatches do
                done = done + inpbatches[i]:size(1)
                yield(done,size,inpbatches[i],tgtbatches[i])
            end
        end
    end
    return wrap(iterator)

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
