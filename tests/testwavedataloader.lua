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
    local tic = torch.tic()
    for s,e,inp,lab in dataloader:sampleiterator(1,nil,true) do
        valuetolab[inp[1][1]] = lab[1]
        numvalues = numvalues + 1
        tester:assert(inp:size(1) == lab:size(1))
    end

    -- Emulate some 10 iterations over the data
    for i=1,10 do
        local tmpnumvalues = numvalues
        for s,e,inp,lab in dataloader:sampleiterator(1,nil,true) do
            if valuetolab[inp[1][1]] then
                if valuetolab[inp[1][1]] == lab[1] then
                    tmpnumvalues = tmpnumvalues - 1
                end
            end
            tester:assert(inp:size(1) == lab:size(1))
        end
        tester:assert(tmpnumvalues == 0,"Error in iteration "..i)
    end
end

if not filelist or filelist == "" then
    print("Please pass a filelist as first argument")
    return
end
tester:add(modeltester)
tester:run()
