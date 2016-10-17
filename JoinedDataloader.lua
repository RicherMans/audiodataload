local adl = require 'audiodataload._base'
local JoinedDataloader = torch.class('adl.JoinedDataloader', 'adl.BaseDataloader',adl)

local initcheck = argcheck{
    pack=true,
    {
        name='module',
        type='adl.BaseDataloader',
        help="The module to wrap around, if nil is passed we assume the data was already preprocessed",
        opt=true,
    },
    {
        name='dirpath',
        type='string',
        help="The directory where the parts are dumped to",
        opt=true
    },
    {
        name='loadfile',
        type='string',
        help="A list of files which is loaded into the Iterator",
        opt=true
    },
    {
        name='cachesize',
        type='number',
        help="If newfile is provided, this arguments specifies the average chunksize for a deflated file, thus also enabling deflation.",
        default=10000,
    }

}


local function dump(cache,targets,outputfile)
    local datatensor = torch.concat(cache,1)
    local targettensor = torch.concat(targets,1)
    local tensor = {
        data = datatensor,
        target = targettensor,
    }
    torch.save(outputfile,tensor)
end

function JoinedDataloader:__init(...)
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end

    -- In case the module is passed and the dirpath is not empty
    if self.dirpath ~= nil and not paths.dirp(self.dirpath) then
        for k,v in pairs(self.module) do self[k] = v end
        paths.mkdir(self.dirpath)
        local size = self.cachesize
        self.doshuffle = false
        local dumps = {}
        local runningid = 1
        local outputfileid = 1
        local outputfile
        local datacounter = 1
        local datacache = {}
        local targetcache = {}
        -- Shuffle the utterances
        for done,finished,input,target,path in self.module:uttiterator(nil,nil,true) do
            path = readfilelabel(path)
            runningid = runningid + input:size(1)
            datacache[datacounter] = input:clone()
            -- Adjust the sizes of the input and the target, since target is only a 1 dim tensor
            targetcache[datacounter] = target:expand(input:size(1),target:size(2)):clone()
            datacounter = datacounter + 1
            if runningid > size then
                outputfile=paths.concat(self.dirpath,"dump_part_"..outputfileid..".th")
                dumps[#dumps+1] = outputfile
                outputfileid = outputfileid + 1
                runningid = 1
                dump(datacache,targetcache,outputfile)
                datacounter = 1
                targetcache = {}
                datacache={}
            end
        end
        -- Dump the rest of the data
        if next(datacache) ~= nil then
            outputfile=paths.concat(self.dirpath,"dump_part_"..outputfileid..".th")
            dump(datacache,targetcache,outputfile)
            dumps[#dumps+1] = outputfile
        end
        self.dumps = dumps

    else
        -- Load from the preexisting cache
        self.dumps = {}
        self.module = {}
        local samplesize,dim = 0,0,0
        local sample,target
        local ntargets
        local datapiece
        for file in io.lines(self.loadfile) do
            self.dumps[#self.dumps + 1] = file
            datapiece = torch.load(file)
            sample = datapiece.data
            target = datapiece.target
            samplesize = samplesize + sample:size(1)
            dim = sample:size(2)
            local  a = torch.LongTensor(target:size():fill(-1))
            ntargets = ntargets or torch.LongTensor(target:size()):fill(-1)
            ntargets:cmax(target:max(1)[1])
        end
        self.module.dim = function()
            return dim
        end
        self.module.usize = function()
            return samplesize
        end
        self.module.size = function()
            return samplesize
        end
        self.module.nClasses = function()
            return ntargets
        end
    end
end

local function batchfeat(feats,targets,size)
    return
        feats:split(size,1),
        targets:split(size,1)
end

function JoinedDataloader:shuffle()
    self.doshuffle = true
end

function JoinedDataloader:sampleiterator(batchsize,epochsize,...)
    batchsize = batchsize or 16
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:size()

    local min = math.min

    local wrap = coroutine.wrap
    local yield = coroutine.yield

    self:beforeIter(unpack(dots))
    -- Buffers
    local fname,size = nil,0
    local input,target = torch.Tensor(),torch.LongTensor()

    local ndumps = #self.dumps
    local dumpids = torch.LongTensor()
    -- Shuffle the order of dumps and the content of the dumps
    if self.doshuffle then
        dumpids = dumpids:randperm(ndumps)
    else
        dumpids = dumpids:range(1,ndumps)
    end

    local function loaddataiter()
        for i=1,dumpids:size(1) do
            local fname = self.dumps[dumpids[i]]
            if fname == nil then return end
            local function loaddata(datafname)
                local data = torch.load(datafname)
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
                collectgarbage()
                return size,inputbatches,targetbatches
            end

            yield(loaddata(fname))
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


function JoinedDataloader:loadAudioUtterance(audiofilepath,wholeutt)
    error("Not implemented")
end

-- Number of utterances in the dataset
-- Just wrap it around the moduel
function JoinedDataloader:usize()
    return self.module:usize()
end

-- Returns the sample size of the dataset
function JoinedDataloader:size()
    return self.module:size()
end

-- Returns the data dimensions
function JoinedDataloader:dim()
    return self.module:dim()
end

function JoinedDataloader:nClasses()
    return self.module:nClasses()
end
