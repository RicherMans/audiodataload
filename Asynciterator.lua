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

function Asynciterator:__init(...)
    local args = initcheck(...)
    for k,v in pairs(args) do
        self[k] = v
    end
    self.recvqueue = torchx.Queue()
    self.ninprogress = 0

    self.module.sampletofeatid, self.module.sampletoclassrange = self.module:sampletofeat(self.module.samplelengths)

    local threads = require 'threads'
    -- Threads share the given tensors
    threads.Threads.serialization('threads.sharedserialize')

    local mainSeed = os.time()
     -- build a Threads pool, or use the last one initilized
    self.threads = threads.Threads(
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
            local success, err = pcall(function()
                _t = {}
                _t.id = idx
                local seed = mainSeed + idx
                math.randomseed(seed)
                torch.setdefaulttensortype('torch.FloatTensor')
                torch.manualSeed(seed)
                if self.verbose then
                    print(string.format('Starting worker thread with id: %d seed: %d memory usage: %d mb', idx, seed,collectgarbage("count")/1024))
                end
                _t.module = self.module
            end)
            if not success then
                error(err)
            end
        end
    )
end

function Asynciterator:forceCollectGarbage()
    self.threads:synchronize()
    self.threads:specific(true)
    -- Collect garbage in all worker threads
    for threadId=1,self.nthreads do
        self.threads:addjob(threadId, function()
            collectgarbage()
        end)
    end
    self.threads:synchronize()
    self.threads:specific(false)
    -- Collect garbage in main thread
    collectgarbage()
end


function Asynciterator:collectgarbage()
    self.gcdelay = self.gcdelay or 50
    self.gccount = (self.gccount or 0) + 1
    if self.gccount >= self.gcdelay then
        self:forceCollectGarbage()
        self.gccount = 0
    end
end

function Asynciterator:synchronize()
    self.threads:synchronize()
    while not self.recvqueue:empty() do
        self.recvqueue:get()
    end
    self.ninprogress = 0
    self:forceCollectGarbage()
end

function Asynciterator:_putQuene(func,args,size)
    assert(torch.type(func) == 'string')
    assert(torch.type(args) == 'table')
    assert(torch.type(size) == 'number') -- size of batch
    for i=1,1000 do
        if self.threads:acceptsjob() then
            break
        else
            sys.sleep(0.01)
        end
        if i==1000 then
            error"infinite loop"
        end
    end

    self.ninprogress = (self.ninprogress or 0) + 1

    self.threads:addjob(
        -- the job callback (runs in data-worker thread)
        function()
            -- func, args and size are upvalues
            local res = {_t.module[func](_t.module,unpack(args))}
            res.size = size
            local start,stop = unpack(args)
            res.start = start
            res.stop=stop
            return res
        end,
        -- the endcallback (runs in the main thread)
        function(res)
            assert(torch.type(res) == 'table')
            self.recvqueue:put(res)
            self.ninprogress = self.ninprogress - 1
        end
    )

end

-- recv results from worker : get results from queue
function Asynciterator:asyncGet()
    -- necessary because Threads:addjob sometimes calls dojob...
    if self.recvqueue:empty() then
        self.threads:dojob()
    end
    assert(not self.recvqueue:empty())
    return self.recvqueue:get()
end


function Asynciterator:sampleiterator(batchsize, epochsize, random,...)
    batchsize = batchsize or 16
    local dots = {...}
    epochsize = epochsize or -1
    epochsize = epochsize > 0 and epochsize or self:size()

    random = random or false

    -- Randomized ids, passed to cacheiterator
    local sampleids = torch.LongTensor()
    if random then
        -- Shuffle the sample list
        -- Apply the randomization
        sampleids = sampleids:randperm(self:size())
    else
        sampleids = sampleids:range(1,self:size())
    end

    local min = math.min

    local nput = 1 -- currently in queune
    local nget = 1 -- overall sampled
    local inputs, targets

    self:beforeIter(unpack(dots))

    local putmode = true
    local stop
    local start = 1
    -- build iterator
    local iterate = function()
        -- finish if
        if nget > epochsize then
            self:afterIter(unpack(dots))
            return
        end
        if nput <= epochsize then
            local bs = min(nput+batchsize , epochsize + 1 ) - nput

            stop = start + bs - 1
            -- print("Starting with ",start,stop,"size:",bs)
            -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
            self:_putQuene('subSamples',{sampleids[{{start,stop}}], unpack(dots)},bs)
            -- -- allows reuse of inputs and targets buffers for next iteration
            -- inputs, targets = batch[1], batch[2]
            nput = nput + bs
            start = start + bs
            if start > self:size() then
                start = 1
            end
        end
        if not putmode then
            local batch = self:asyncGet()
            --  -- we will resend these buffers to the workers next call
            nget = nget + batch.size
            self:collectgarbage()
            -- print(batch.start,batch.stop,nget,nput,epochsize)
            return nget - 1 , epochsize, unpack(batch)
        end
        return
    end
    for thread=1,self.nthreads do
        iterate()
    end
    putmode = false
    return iterate
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
