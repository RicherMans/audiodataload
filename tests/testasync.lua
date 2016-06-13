require 'torch'
require 'nn'
require 'xlua'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testasync",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()
ProFi = require 'ProFi'



function modeltester:init()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)
    local asyncdata = audioload.Asynciterator(dataloader)
end


function modeltester:testsampleiteratormultiple()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)
    local asyncdata = audioload.Asynciterator(dataloader,4)

    for s,e,inp,lab in dataloader:sampleiterator(128) do
        tester:assert(inp:size(1) == lab:size(1))
    end

    -- Multiple loops
    for s,e,inp,lab in asyncdata:sampleiterator(128) do
        tester:assert(inp:size(1) == lab:size(1))
    end

    for s,e,inp,lab in asyncdata:sampleiterator(128) do
        tester:assert(inp:size(1) == lab:size(1))
    end
end

function modeltester:testsampleiteratorsinglethread()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)
    local asyncdata = audioload.Asynciterator(dataloader,1)

    for s,e,inp,lab in asyncdata:sampleiterator(128) do
        tester:assert(inp:size(1) == lab:size(1))
    end

end

tester:add(modeltester)
tester:run()
