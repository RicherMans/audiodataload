local adl = require 'audiodataload._base'
local Asynciterator = torch.class('adl.Asynciterator', 'adl.BaseDataloader', adl)

local initcheck = argcheck{
    pack=true,
    {
        name='module',
        type='adl.BaseDataloader',
        help="The module to wrap around",
    },
    {
        name = "nthreads",
        type= "number",
        help="Number of threads",
        default = 1,
        check= function(num)
            if num > 0 then return true else return false end
        end
    },
    {
        name = 'verbose',
        type = "boolean",
        help="Prints verbose messages",
        default=false
    },
    {
        name="serialmode",
        type='string',
        check=function(mode)
            if mode == 'binary' or mode == 'ascii' then return true else return false end
        end,
        default="binary"
    }
}
local Threads = require 'threads'
-- Threads share the given tensors
Threads.serialization('threads.sharedserialize')

function Asynciterator:__init(...)

    local args = initcheck(...)
    for k,v in pairs(args) do
        self[k] = v
    end
    self.recvqueue = torchx.Queue()
    self.ninprogress = 0



    local modstr = torch.serialize(self.module)

    self.sampleids = self.module.sampleids
    self.uttids = self.module.uttids

    local mainSeed = os.time()
     -- build a Threads pool, or use the last one initilized
    self.threads = Threads(
        self.nthreads, -- the following functions are executed in each thread
        function()
            -- For wavedataloader
            audio = audio or require 'audio'
            -- For HtkDataloader
            _htktorch = _htktorch or require 'torchhtk'
            -- For others
            audiodataload = audiodataload or require "audiodataload"
        end,
        function(idx)
                torch.setnumthreads(1)
                _t = {}
                _t.id = idx
                local seed = mainSeed + idx
                torch.setdefaulttensortype('torch.FloatTensor')
                torch.manualSeed(seed)
                if self.verbose then
                    print(string.format('Starting worker thread with id: %d seed: %d memory usage: %d mb', idx, seed,collectgarbage("count")/1024))
                end
                _t.module = torch.deserialize(modstr)
        end
    )
end





function Asynciterator:sampleiterator(batchsize, epochsize, ...)
    batchsize = batchsize or 16
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:size()


    self.sampletofeatid, self.sampletoclassrange = self.module:sampletofeat(self.module.samplelengths)

    local min = math.min

    local nput = 1 -- currently in queune
    local nget = 1 -- overall sampled

    self:beforeIter(unpack(dots))

    local putmode = true
    local stop = 0
    local start = 1
    -- Return varibales for the threads
    local input = nil
    local restarget = nil
    local itersize = nil
    -- Buffer variables for the threads
    local mthreads = self.threads
    local sampleids = self.sampleids
    local sampletofeatid = self.sampletofeatid
    local sampletoclassrange = self.sampletoclassrange
    local filelabels = self.module.filelabels
    local targets = self.module.targets

    local function runthread()
        while start <= epochsize and mthreads:acceptsjob() do
            local bs = min(start+batchsize , epochsize + 1 ) - start

            stop = start + bs - 1
            -- Need to keep these variables local for the threads
            local cursampleids = sampleids[{{start,stop}}]
            -- the row from the file lists
            local featids = sampletofeatid:index(1,cursampleids)
            -- The range of the current sample within the class
            local classranges = sampletoclassrange:index(1,cursampleids)
            -- Labels are passed to the getsample to obtain the datavector
            local labels = filelabels:index(1,featids)
            local target = targets:index(1,featids)
            mthreads:addjob(
                function(labels,classranges)
                    local res = {}
                    res.input = _t.module["getSample"](_t.module,labels,classranges)
                    res.target = target
                    res.size = target:size(1)
                    collectgarbage()
                    return res
                end,
                function(tab)
                    input = tab.input
                    restarget = tab.target:view(tab.target:nElement())
                    itersize = tab.size
                end,
                labels,
                classranges
            )
            start = start + batchsize
        end
    end

    local finishedcount = 0
    local function iterate()
        runthread()
        if not mthreads:hasjob() then
            return nil
        end
        mthreads:dojob()
        if mthreads:haserror() then
            mthreads:synchronize()
        end
        if not input or not restarget then
            -- mthreads:dojob()
            mthreads:synchronize()
        end
        finishedcount = finishedcount + itersize
        return finishedcount, epochsize, input,restarget
    end
    return iterate
end

function Asynciterator:reset()
    -- self:collectgarbage()
    -- self.recvqueue = self.recvqueue.new()
    -- self.threads:synchronize()
    -- -- Finish the job
    -- while not self.recvqueue:empty() do
    --     self:asyncGet()
    -- end
end

function Asynciterator:nClasses()
    return self.module:nClasses()
end

function Asynciterator:usize()
    return self.module:usize()
end

function Asynciterator:dim()
    return self.module:dim()
end

function Asynciterator:size()
    return self.module:size()
end
