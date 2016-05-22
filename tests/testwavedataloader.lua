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


function modeltester:init()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)
end

function modeltester:testSeqlenSample()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100,seqlen=5,padding='right'}

    local it = dataloader:sampleiterator(128)
    for i,k,v,t in it do
        tester:assert(i ~= nil)
        tester:assert(k ~= nil)
        tester:assert(v ~= nil)
        tester:assert(t ~= nil)
        -- xlua.progress(i,k)
        -- print(i,k,v,d)
    end
end

function modeltester:testUtteranceSeq()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100,seqlen=5,padding='right'}

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
    print("Took "..torch.toc(tic))
end
function modeltester:testUtteranceNoSeq()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100,seqlen=1,padding='right'}

    local tic = torch.tic()
    local it = dataloader:uttiterator(1)
    for i,k,v,t in it do
        print(i,k)
        tester:assert(i ~= nil)
        tester:assert(k ~= nil)
        tester:assert(v:nonzero() ~= nil)
        tester:assert(v ~= nil)
        tester:assert(t ~= nil)
        -- xlua.progress(i,k)
        -- print(i,k,v,d)
    end
    print("Took "..torch.toc(tic))
end

tester:add(modeltester)
tester:run()
