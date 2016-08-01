require 'torch'
require 'nn'
require 'xlua'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testwavedataloader",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()
ProFi = require 'ProFi'


local filelist = arg[1]
function modeltester:init()
    local filepath = filelist
    local dataloader = audioload.WaveDataloader(filepath,100)
end

local function calcnumframes(samplessize,framesize,shift)
    return math.floor((samplessize-framesize)/shift + 1)
end

function modeltester:testrandomize()
    local filepath = filelist
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100}
    ProFi:start()
    local it = dataloader:sampleiterator(128,nil,true)
    for i,k,v,t in it do
        -- tester:assert(i ~= nil)
        -- tester:assert(k ~= nil)
        tester:assert(v ~= nil)
        tester:assert(t ~= nil)
    end
    ProFi:stop()
    ProFi:writeReport('testrandomize.txt')
end

function modeltester:testUtteranceSeq()
    local filepath = filelist
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100}

    local it = dataloader:uttiterator(1)
    local tic = torch.tic()
    for i,k,v,t in it do
        tester:assert(i ~= nil)
        tester:assert(k ~= nil)
        tester:assert(v:nonzero() ~= nil)
        tester:assert(v ~= nil)
        tester:assert(t ~= nil)
        -- xlua.progress(i,k)
    end
end

function modeltester:testUtteranceNoSeq()
    local filepath = filelist
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100}

    local tic = torch.tic()
    ProFi:start()
    local it = dataloader:uttiterator()
    for i,k,v,t in it do
        tester:assert(i ~= nil)
        tester:assert(k ~= nil)
        tester:assert(v:nonzero() ~= nil)
        tester:assert(v ~= nil)
        tester:assert(t ~= nil)
        -- xlua.progress(i,k)
        -- print(i,k,v,d)
    end
    ProFi:stop()
    ProFi:writeReport('testutterancenoseq.txt')
end

function modeltester:testRandomize()
    local filepath = filelist
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100}
    valuetolab={}
    numvalues = 0
    local classsizes= torch.Tensor(dataloader:nClasses()):zero()
    local sampletoclass = torch.Tensor(dataloader:size())
    for s,e,inp,lab in dataloader:sampleiterator(1,nil) do
        valuetolab[inp[1][1]] = lab[1]
        numvalues = numvalues + 1
        tester:assert(inp:size(1) == lab:size(1))
        sampletoclass[s] = lab[1]
        local addone = torch.Tensor(lab:size(1)):fill(1)
        classsizes:indexAdd(1,lab:long(),addone)
    end
    -- Check if we only have unique values 
    local uniquevalues = 0
    for k,v in pairs(valuetolab) do
        uniquevalues = uniquevalues + 1
    end

    if uniquevalues ~= numvalues then 
        print("Error, values are not unique. Just a warning, nothing special.  ".. uniquevalues .."/"..numvalues)
    end
    tester:assert(classsizes:sum()==dataloader:size())

    local randomizetol = math.ceil(dataloader:size()/2)
    -- Emulate some 5 iterations over the data
    for i=1,5 do
        local randomized = 0
        local tmpclasssizes = torch.Tensor(dataloader:nClasses()):zero()
        for s,e,inp,lab in dataloader:sampleiterator(1,nil,true) do
            if sampletoclass[s] ~= lab[1] then
                randomized = randomized + 1
            end
            local addone = torch.Tensor(lab:size(1)):fill(1)
            tmpclasssizes:indexAdd(1,lab:long(),addone)
            tester:assert(inp:size(1) == lab:size(1))
        end
        tester:eq(tmpclasssizes,classsizes)
        tester:assert(randomized ~= 0 and randomized > randomizetol,"Randomization not done correctly")
    end
end

function modeltester:testnonrandomizedsamples()
    local filepath = filelist
    local dataloader = audioload.WaveDataloader(filepath,100)
    local classsizes= torch.Tensor(dataloader:nClasses()):zero()


    local batchsize = 128
    for s,e,k,v in dataloader:sampleiterator(batchsize) do
        local addone = torch.Tensor(v:size(1)):fill(1)
        classsizes:indexAdd(1,v:long(),addone)
    end
    tester:assert(classsizes:sum()==dataloader:size())
    for i=1,3 do
        local tmpclasssizes = torch.Tensor(dataloader:nClasses()):zero()
        for s,e,k,v in dataloader:sampleiterator(batchsize) do
            local addone = torch.Tensor(v:size(1)):fill(1)
            tmpclasssizes:indexAdd(1,v:long(),addone)
        end
        tester:eq(tmpclasssizes,classsizes)
    end

end

if not filelist or filelist == "" then
    print("Please pass a filelist as first argument")
    return
end
tester:add(modeltester)
tester:run()
