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
        default = 2
    }
}

function Asynciterator:__init(...)
    local args = initcheck(...)
    for k,v in pairs(args) do
        self[k] = v
    end
    -- Using torch Threads
    local threads = threads or require 'threads'
    -- Threads share the given tensors
    threads.Threads.serialization('threads.sharedserialize')
    local mainSeed = os.time()
    -- build a Threads pool
    self.threads = threads.Threads(
        self.nthreads, -- the following functions are executed in each thread
        function()
        end,
        function(idx)
            local success, err = pcall(function()
                local seed = mainSeed + idx
                math.randomseed(seed)
                torch.manualSeed(seed)
                if verbose then
                    print(string.format('Starting worker thread with id: %d seed: %d', idx, seed))
                end
            end)
            if not success then
                error(err)
            end
        end
    )
    self.recvqueue = torchx.Queue()
    self.ninprogress = 0
end

function Asynciterator:forceCollectGarbage()
    self.threads:synchronize()
    self.threads:specific(true)
    -- Collect garbage in all worker threads
    for threadId=1,self.nthread do
        self.threads:addjob(threadId, function()
            collectgarbage()
        end)
    end
    self.threads:synchronize()
    self.threads:specific(false)
    -- Collect garbage in main thread
    collectgarbage()
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
    assert(torch.type(fn) == 'string')
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
            local success, res = pcall(function()
                -- fn, args and size are upvalues
                local res = {self.module[func](unpack(args))}
                res.size = size
                return res
            end)
            if not success then
                error(res)
            end
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
function AsyncIterator:asyncGet()
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
    epochsize = epochsize > 0 and epochsize or self:size()

    random = random or false
    if not ( self.sampletofeatid and self.sampletoclassrange ) then
        self.sampletofeatid,self.sampletoclassrange = self.module:sampletofeat(self.samplelengths)
    end

    if random then
        -- Shuffle the list
        local randomids = torch.LongTensor():randperm(self:size())

        self.sampletofeatid = self.sampletofeatid:index(1,randomids)
        self.sampletoclassrange = self.sampletoclassrange:index(1,randomids)
    end

    local min = math.min

    local nput = 0 -- currently in queune
    local nget = 0 -- overall sampled
    local inputs, targets

    self._start = self._start or 1 -- Start index for multiple threads

    self:beforeIter(unpack(dots))

    local putmode = true

    -- build iterator
    local iterate = function()
        -- finish if
        if nget > epochsize then
            self.module:afterIter(unpack(dots))
            return
        end
        if nput < epochsize then
            local bs = min(nput+batchsize, epochsize + 1) - nput

            local stop = min(self._start + bs - 1,size)
            -- Sequence length is via default not used, thus returns an iterator of size Batch X DIM
            local batch = {self:_putQuene('subSamples',{self._start, stop, random, unpack(dots)})}
            -- -- allows reuse of inputs and targets buffers for next iteration
            -- inputs, targets = batch[1], batch[2]

            nput = nput + bs
            if self._start >= size then
                self._start = 1
            end
        end
        if not putmode then
            local batch = self:asyncGet()
            --  -- we will resend these buffers to the workers next call
            --  previnputs, prevtargets = batch[1], batch[2]
            nget = nget + batch.size
            self:collectgarbage()
            return nget, unpack(batch)
        end
        return nget - 1,epochsize, unpack(batch)
    end
    for thread=1,self.thread do
        iterate()
    end
    putmode = false
    return iterate
end

function Asynciterator:size()
   return self.module:size()
end
