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

    local it = dataloader:sampleiterator()
    dataloader:random()
    -- for i,k,v,d in it do
    --     xlua.progress(i,k)
    --     -- print(i,k,v,d)
    -- end
end

function modeltester:testSeqlenSample()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100,seqlen=100,padding='right'}

    local it = dataloader:sampleiterator()
    for i,k,v,t in it do
        tester:assert(i ~= nil)
        tester:assert(k ~= nil)
        tester:assert(v ~= nil)
        tester:assert(t ~= nil)
        -- xlua.progress(i,k)
        -- print(i,k,v,d)
    end
end

function modeltester:testUtterance()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100,seqlen=100,padding='right'}

    local it = dataloader:uttiterator()
    for i,k,v,t in it do
        tester:assert(i ~= nil)
        tester:assert(k ~= nil)
        tester:assert(v ~= nil)
        tester:assert(t ~= nil)
        -- xlua.progress(i,k)
        -- print(i,k,v,d)
    end
end

tester:add(modeltester)
tester:run()
